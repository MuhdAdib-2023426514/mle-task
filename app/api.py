"""Inference-free FastAPI retrieval service backed by PostgreSQL + pgvector."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request
from psycopg_pool import ConnectionPool, PoolTimeout

from app.config import Settings
from app.database import (
    check_database_health,
    create_pool,
    find_industry_fallback_companies,
    find_similar_companies,
    get_company_profile,
)
from app.normalization import normalize_registration_number
from app.schemas import HealthResponse, SimilarCompaniesResponse, SimilarCompanyItem

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5
MAX_LIMIT = 20


def create_app(
    *,
    settings: Settings | None = None,
    db_pool: ConnectionPool[Any] | None = None,
) -> FastAPI:
    """Create the FastAPI application, optionally injecting a DB pool for testing."""
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owns_pool = db_pool is None
        pool = db_pool
        if pool is None:
            # min_size=0 avoids eager startup failures if DB or migrations are not ready yet.
            pool = create_pool(
                cfg.effective_database_url,
                min_size=0,
                max_size=cfg.db_pool_max_size,
            )
        application.state.db_pool = pool
        try:
            yield
        finally:
            if owns_pool:
                pool.close()

    application = FastAPI(
        title="OneCredit Similar Companies API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pool: ConnectionPool[Any] = request.app.state.db_pool
        try:
            with pool.connection(timeout=3.0) as conn:
                ok = check_database_health(conn)
        except (psycopg.Error, PoolTimeout, OSError) as exc:
            logger.error("Health check failed: %s: %s", type(exc).__name__, exc)
            raise HTTPException(status_code=503, detail="Database unavailable") from exc
        if not ok:
            raise HTTPException(status_code=503, detail="Database unavailable")
        return HealthResponse(status="ok", database="ok")

    @application.get(
        "/companies/{registration_number}/similar",
        response_model=SimilarCompaniesResponse,
        response_model_by_alias=True,
    )
    def get_similar_companies(
        registration_number: str,
        request: Request,
        limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> SimilarCompaniesResponse:
        try:
            canonical_id = normalize_registration_number(registration_number)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        pool: ConnectionPool[Any] = request.app.state.db_pool
        try:
            with pool.connection(timeout=5.0) as conn:
                profile = get_company_profile(conn, canonical_id)
                if profile is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Company not found: {canonical_id}",
                    )
                if not profile.has_embedding:
                    if profile.industry is None:
                        return SimilarCompaniesResponse(
                            registration_number=profile.registration_number,
                            company_name=profile.company_name,
                            industry=profile.industry,
                            state=profile.state,
                            status=profile.status,
                            recommendations=[],
                            message=(
                                "No business description or industry metadata "
                                "available for this company."
                            ),
                        )
                    fallback_rows = find_industry_fallback_companies(
                        conn,
                        canonical_id,
                        industry=profile.industry,
                        state=profile.state,
                        limit=limit,
                    )
                    fallback_items = [
                        SimilarCompanyItem(
                            registration_number=row.registration_number,
                            company_name=row.company_name,
                            industry=row.industry,
                            state=row.state,
                            similarity=None,
                            match_type=row.match_type,
                        )
                        for row in fallback_rows
                    ]
                    fallback_msg = (
                        "No business description available; showing same-industry fallback matches."
                        if fallback_items
                        else (
                            "No business description available and no active "
                            "same-industry companies found."
                        )
                    )
                    return SimilarCompaniesResponse(
                        registration_number=profile.registration_number,
                        company_name=profile.company_name,
                        industry=profile.industry,
                        state=profile.state,
                        status=profile.status,
                        recommendations=fallback_items,
                        message=fallback_msg,
                    )
                neighbors = find_similar_companies(conn, canonical_id, limit=limit)
        except HTTPException:
            raise
        except (psycopg.Error, PoolTimeout, OSError) as exc:
            logger.error(
                "Database error querying similar companies for %s: %s: %s",
                canonical_id,
                type(exc).__name__,
                exc,
            )
            raise HTTPException(status_code=503, detail="Database unavailable") from exc

        items = [
            SimilarCompanyItem(
                registration_number=row.registration_number,
                company_name=row.company_name,
                industry=row.industry,
                state=row.state,
                similarity=row.similarity,
                match_type=row.match_type,
            )
            for row in neighbors
        ]
        message = None if items else "No similar active companies found."
        return SimilarCompaniesResponse(
            registration_number=profile.registration_number,
            company_name=profile.company_name,
            industry=profile.industry,
            state=profile.state,
            status=profile.status,
            recommendations=items,
            message=message,
        )

    return application


app = create_app()
