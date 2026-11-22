"""API tests against isolated Postgres/pgvector and simulated DB outage."""

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from app.api import create_app
from app.database import mark_company_struck_off, upsert_company
from app.normalization import compute_event_version_key
from app.schemas import CompanyStatus, StruckOffCompanyEvent, UpsertCompanyEvent

pytestmark = pytest.mark.integration


def _unit_vec(v0: float, v1: float, dim: int = 384) -> list[float]:
    """Create a 384-dim vector in the first two coordinates for exact cosine tests."""
    return [v0, v1] + [0.0] * (dim - 2)


def _seed_company(
    pool: ConnectionPool[Any],
    *,
    reg: str,
    name: str | None,
    industry: str | None,
    state: str | None,
    desc: str | None,
    embedding: list[float] | None,
    struck_off: bool = False,
) -> None:
    ts = datetime(2026, 8, 14, 10, 0, 0, tzinfo=UTC)
    event = UpsertCompanyEvent(
        registration_number=reg,
        company_name=name,
        industry=industry,
        state=state,
        nature_of_business=desc,
        updated_at=ts,
        raw_updated_at=ts.isoformat(),
        event_version_key=compute_event_version_key(
            event_type="company-upserted",
            registration_number=reg,
            status=CompanyStatus.ACTIVE,
            company_name=name,
            industry=industry,
            state=state,
            nature_of_business=desc,
        ),
        message_id=f"seed-{reg}",
    )
    with pool.connection() as conn:
        with conn.transaction():
            upsert_company(
                conn,
                event,
                embedding=embedding,
                embedding_model="test-model" if embedding is not None else None,
                embedding_text_hash=f"hash-{reg}" if embedding is not None else None,
            )
            if struck_off:
                so_ts = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)
                mark_company_struck_off(
                    conn,
                    StruckOffCompanyEvent(
                        registration_number=reg,
                        updated_at=so_ts,
                        raw_updated_at=so_ts.isoformat(),
                        event_version_key=compute_event_version_key(
                            event_type="company-struck-off",
                            registration_number=reg,
                            status=CompanyStatus.STRUCK_OFF,
                        ),
                        message_id=f"so-{reg}",
                    ),
                )


class _BrokenPool:
    """Pool double that always simulates a PostgreSQL outage."""

    @contextmanager
    def connection(self, timeout: float = 5.0) -> Any:
        raise psycopg.OperationalError("connection refused")
        yield  # pragma: no cover

    def close(self) -> None:
        return None


def test_similar_companies_ranks_by_cosine_and_excludes_self_struck_off_and_nulls(
    db_pool: ConnectionPool[Any],
) -> None:
    # Target vector along +x: (1, 0)
    _seed_company(
        db_pool,
        reg="SYN-000001",
        name="Target Cafe Sdn. Bhd.",
        industry="Food & Beverage",
        state="Selangor",
        desc="Specialty coffee shop",
        embedding=_unit_vec(1.0, 0.0),
    )
    # Closest active match: cos = 0.96
    _seed_company(
        db_pool,
        reg="SYN-000002",
        name="Closest Cafe Sdn. Bhd.",
        industry="Food & Beverage",
        state="Johor",
        desc="Cafe and bakery",
        embedding=_unit_vec(0.96, 0.28),
    )
    # Second match: cos = 0.60, with null optional fields preserved
    _seed_company(
        db_pool,
        reg="SYN-000003",
        name="Mid Match Trading",
        industry=None,
        state=None,
        desc="Restaurant supplies",
        embedding=_unit_vec(0.60, 0.80),
    )
    # Orthogonal match: cos = 0.0
    _seed_company(
        db_pool,
        reg="SYN-000004",
        name="Unrelated Logistics PLT",
        industry="Logistics",
        state="Kuala Lumpur",
        desc="Freight trucking",
        embedding=_unit_vec(0.0, 1.0),
    )
    # Struck-off company with identical vector (cos = 1.0) -> MUST be excluded!
    _seed_company(
        db_pool,
        reg="SYN-000005",
        name="Closed Cafe Sdn. Bhd.",
        industry="Food & Beverage",
        state="Selangor",
        desc="Specialty coffee shop",
        embedding=_unit_vec(1.0, 0.0),
        struck_off=True,
    )
    # Company without embedding -> MUST be excluded!
    _seed_company(
        db_pool,
        reg="SYN-000006",
        name="No Desc Sdn. Bhd.",
        industry="Food & Beverage",
        state="Perak",
        desc=None,
        embedding=None,
    )

    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        # Also test lowercase + whitespace normalization in path
        resp = client.get("/companies/%20syn-000001%20/similar?limit=5")
        assert resp.status_code == 200
        body = resp.json()

    assert body["registrationNumber"] == "SYN-000001"
    assert body["companyName"] == "Target Cafe Sdn. Bhd."
    assert body["status"] == "active"
    assert body["message"] is None

    recs = body["recommendations"]
    assert [r["registrationNumber"] for r in recs] == [
        "SYN-000002",
        "SYN-000003",
        "SYN-000004",
    ]
    assert recs[0]["similarity"] == pytest.approx(0.96, abs=1e-4)
    assert recs[0]["matchType"] == "semantic"
    assert recs[1]["similarity"] == pytest.approx(0.60, abs=1e-4)
    assert recs[1]["industry"] is None
    assert recs[1]["state"] is None
    assert recs[2]["similarity"] == pytest.approx(0.0, abs=1e-4)


def test_equal_similarity_ties_prefer_same_state_then_industry(
    db_pool: ConnectionPool[Any],
) -> None:
    _seed_company(
        db_pool,
        reg="SYN-000100",
        name="Source Cafe",
        industry="Food & Beverage",
        state="Selangor",
        desc="Coffee shop",
        embedding=_unit_vec(1.0, 0.0),
    )
    # Earlier ID, different state, same industry
    _seed_company(
        db_pool,
        reg="SYN-000101",
        name="Johor Cafe",
        industry="Food & Beverage",
        state="Johor",
        desc="Coffee shop",
        embedding=_unit_vec(1.0, 0.0),
    )
    # Same state, different industry
    _seed_company(
        db_pool,
        reg="SYN-000102",
        name="Selangor Other",
        industry="Retail",
        state="Selangor",
        desc="Coffee shop",
        embedding=_unit_vec(1.0, 0.0),
    )
    # Later ID, same state AND same industry -> must rank #1 among 1.0 ties
    _seed_company(
        db_pool,
        reg="SYN-000103",
        name="Selangor Cafe",
        industry="Food & Beverage",
        state="Selangor",
        desc="Coffee shop",
        embedding=_unit_vec(1.0, 0.0),
    )

    # Candidate with NULL state must NOT sort ahead of same-state match (Postgres NULLS FIRST trap)
    _seed_company(
        db_pool,
        reg="SYN-000104",
        name="Null State Cafe",
        industry="Food & Beverage",
        state=None,
        desc="Coffee shop",
        embedding=_unit_vec(1.0, 0.0),
    )

    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        resp = client.get("/companies/SYN-000100/similar?limit=5")
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert [r["registrationNumber"] for r in recs] == [
        "SYN-000103",
        "SYN-000102",
        "SYN-000101",
        "SYN-000104",
    ]
    assert all(r["matchType"] == "semantic" for r in recs)


def test_unknown_company_returns_404(db_pool: ConnectionPool[Any]) -> None:
    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        resp = client.get("/companies/SYN-999999/similar")
    assert resp.status_code == 404
    assert "SYN-999999" in resp.json()["detail"]


def test_company_without_embedding_and_with_industry_returns_labeled_fallback(
    db_pool: ConnectionPool[Any],
) -> None:
    _seed_company(
        db_pool,
        reg="SYN-000014",
        name="No Description Hotel Sdn. Bhd.",
        industry="Hospitality",
        state="Kuala Lumpur",
        desc=None,
        embedding=None,
    )
    _seed_company(
        db_pool,
        reg="SYN-000015",
        name="Null State Hotel",
        industry="Hospitality",
        state=None,
        desc="Boutique hotel",
        embedding=_unit_vec(1.0, 0.0),
    )
    _seed_company(
        db_pool,
        reg="SYN-000016",
        name="KL Hotel",
        industry="Hospitality",
        state="Kuala Lumpur",
        desc="Boutique hotel in Kuala Lumpur",
        embedding=_unit_vec(0.0, 1.0),
    )
    _seed_company(
        db_pool,
        reg="SYN-000017",
        name="KL Logistics",
        industry="Logistics",
        state="Kuala Lumpur",
        desc="Parcel delivery",
        embedding=_unit_vec(1.0, 0.0),
    )
    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        resp = client.get("/companies/SYN-000014/similar?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["registrationNumber"] == "SYN-000014"
    assert "fallback" in (body["message"] or "").lower()
    recs = body["recommendations"]
    # Same-state peer ranks first, Logistics excluded, similarity is never fabricated (null)
    assert [r["registrationNumber"] for r in recs] == ["SYN-000016", "SYN-000015"]
    assert all(r["matchType"] == "industry_fallback" for r in recs)
    assert all(r["similarity"] is None for r in recs)


def test_company_without_embedding_or_industry_returns_empty_list(
    db_pool: ConnectionPool[Any],
) -> None:
    _seed_company(
        db_pool,
        reg="SYN-000018",
        name="Incomplete Co Sdn. Bhd.",
        industry=None,
        state="Selangor",
        desc=None,
        embedding=None,
    )
    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        resp = client.get("/companies/SYN-000018/similar?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendations"] == []
    assert body["message"] is not None


def test_company_with_no_active_candidates_returns_empty_list(
    db_pool: ConnectionPool[Any],
) -> None:
    _seed_company(
        db_pool,
        reg="SYN-000077",
        name="Only Active Company",
        industry="Technology",
        state="Penang",
        desc="Cloud engineering",
        embedding=_unit_vec(1.0, 0.0),
    )
    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        resp = client.get("/companies/SYN-000077/similar?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendations"] == []
    assert body["message"] == "No similar active companies found."


def test_struck_off_source_can_still_be_viewed_and_gets_active_peers(
    db_pool: ConnectionPool[Any],
) -> None:
    _seed_company(
        db_pool,
        reg="SYN-000080",
        name="Struck Off Co",
        industry="Retail",
        state="Selangor",
        desc="Mobile phone retail",
        embedding=_unit_vec(1.0, 0.0),
        struck_off=True,
    )
    _seed_company(
        db_pool,
        reg="SYN-000081",
        name="Active Peer Co",
        industry="Retail",
        state="Johor",
        desc="Mobile phone shop",
        embedding=_unit_vec(1.0, 0.0),
    )
    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        resp = client.get("/companies/SYN-000080/similar?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "struck_off"
    assert [r["registrationNumber"] for r in body["recommendations"]] == ["SYN-000081"]


def test_limit_validation_and_blank_registration_number(db_pool: ConnectionPool[Any]) -> None:
    _seed_company(
        db_pool,
        reg="SYN-000090",
        name="A",
        industry="Retail",
        state="Perak",
        desc="Retail shop",
        embedding=_unit_vec(1.0, 0.0),
    )
    _seed_company(
        db_pool,
        reg="SYN-000091",
        name="B",
        industry="Retail",
        state="Perak",
        desc="Retail store",
        embedding=_unit_vec(0.9, 0.43589),
    )
    _seed_company(
        db_pool,
        reg="SYN-000092",
        name="C",
        industry="Retail",
        state="Perak",
        desc="Retail kiosk",
        embedding=_unit_vec(0.8, 0.6),
    )
    app = create_app(db_pool=db_pool)
    with TestClient(app) as client:
        assert (
            len(client.get("/companies/SYN-000090/similar?limit=1").json()["recommendations"]) == 1
        )
        assert client.get("/companies/SYN-000090/similar?limit=0").status_code == 422
        assert client.get("/companies/SYN-000090/similar?limit=21").status_code == 422
        assert client.get("/companies/SYN-000090/similar?limit=abc").status_code == 422
        assert client.get("/companies/%20%20/similar").status_code == 422


def test_health_and_database_outage_return_503(db_pool: ConnectionPool[Any]) -> None:
    healthy_app = create_app(db_pool=db_pool)
    with TestClient(healthy_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "database": "ok"}

    broken_app = create_app(db_pool=_BrokenPool())  # type: ignore[arg-type]
    with TestClient(broken_app) as client:
        assert client.get("/health").status_code == 503
        resp = client.get("/companies/SYN-000001/similar")
        assert resp.status_code == 503
        assert resp.json() == {"detail": "Database unavailable"}
