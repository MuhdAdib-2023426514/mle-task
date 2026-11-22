"""PostgreSQL + pgvector schema migration and version-safe persistence helpers."""

import logging
from pathlib import Path
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.schemas import (
    CompanyProfile,
    CompanyStatus,
    MatchType,
    QuarantineRecord,
    SimilarCompanyRow,
    StoredCompanyVersion,
    StruckOffCompanyEvent,
    UpsertCompanyEvent,
)

logger = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_VERSION_WINS_EXPR = """(
    (companies.updated_at IS NULL AND EXCLUDED.updated_at IS NOT NULL)
    OR (
        companies.updated_at IS NOT NULL
        AND EXCLUDED.updated_at IS NOT NULL
        AND (
            EXCLUDED.updated_at > companies.updated_at
            OR (
                EXCLUDED.updated_at = companies.updated_at
                AND EXCLUDED.event_version_key > companies.event_version_key
            )
        )
    )
    OR (
        companies.updated_at IS NULL
        AND EXCLUDED.updated_at IS NULL
        AND EXCLUDED.event_version_key > companies.event_version_key
    )
)"""

_VERSION_WINS_WHERE_SQL = f"WHERE {_VERSION_WINS_EXPR}"

_UPSERT_COMPANY_SQL = f"""
INSERT INTO companies (
    registration_number,
    company_name,
    industry,
    state,
    nature_of_business,
    status,
    updated_at,
    raw_updated_at,
    event_version_key,
    last_message_id,
    embedding,
    embedding_model,
    embedding_text_hash,
    ingested_at
)
VALUES (
    %s, %s, %s, %s, %s, 'active', %s, %s, %s, %s, %s, %s, %s, NOW()
)
ON CONFLICT (registration_number) DO UPDATE SET
    company_name = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.company_name
        ELSE COALESCE(companies.company_name, EXCLUDED.company_name)
    END,
    industry = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.industry
        ELSE COALESCE(companies.industry, EXCLUDED.industry)
    END,
    state = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.state
        ELSE COALESCE(companies.state, EXCLUDED.state)
    END,
    nature_of_business = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.nature_of_business
        ELSE COALESCE(companies.nature_of_business, EXCLUDED.nature_of_business)
    END,
    status = CASE
        WHEN {_VERSION_WINS_EXPR} THEN 'active'
        ELSE companies.status
    END,
    updated_at = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.updated_at
        ELSE companies.updated_at
    END,
    raw_updated_at = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.raw_updated_at
        ELSE companies.raw_updated_at
    END,
    event_version_key = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.event_version_key
        ELSE companies.event_version_key
    END,
    last_message_id = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.last_message_id
        ELSE companies.last_message_id
    END,
    embedding = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.embedding
        ELSE COALESCE(companies.embedding, EXCLUDED.embedding)
    END,
    embedding_model = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.embedding_model
        ELSE COALESCE(companies.embedding_model, EXCLUDED.embedding_model)
    END,
    embedding_text_hash = CASE
        WHEN {_VERSION_WINS_EXPR} THEN EXCLUDED.embedding_text_hash
        ELSE COALESCE(companies.embedding_text_hash, EXCLUDED.embedding_text_hash)
    END,
    ingested_at = NOW()
WHERE
    {_VERSION_WINS_EXPR}
    OR (
        companies.status = 'struck_off'
        AND companies.company_name IS NULL
        AND EXCLUDED.company_name IS NOT NULL
    )
RETURNING registration_number
"""

_STRUCK_OFF_COMPANY_SQL = f"""
INSERT INTO companies (
    registration_number,
    status,
    updated_at,
    raw_updated_at,
    event_version_key,
    last_message_id,
    ingested_at
)
VALUES (
    %s, 'struck_off', %s, %s, %s, %s, NOW()
)
ON CONFLICT (registration_number) DO UPDATE SET
    status = 'struck_off',
    updated_at = EXCLUDED.updated_at,
    raw_updated_at = EXCLUDED.raw_updated_at,
    event_version_key = EXCLUDED.event_version_key,
    last_message_id = EXCLUDED.last_message_id,
    ingested_at = NOW()
{_VERSION_WINS_WHERE_SQL}
RETURNING registration_number
"""

_GET_COMPANY_VERSION_SQL = """
SELECT
    registration_number,
    status,
    updated_at,
    event_version_key,
    company_name IS NOT NULL AS has_metadata,
    embedding,
    embedding_model,
    embedding_text_hash
FROM companies
WHERE registration_number = %s
"""

_RECORD_INGESTION_ERROR_SQL = """
INSERT INTO ingestion_errors (
    message_id,
    registration_number,
    error_type,
    error_message,
    raw_payload
)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (message_id, error_type) DO NOTHING
"""

_GET_COMPANY_PROFILE_SQL = """
SELECT
    registration_number,
    company_name,
    industry,
    state,
    nature_of_business,
    status,
    embedding IS NOT NULL AS has_embedding
FROM companies
WHERE registration_number = %s
"""

_FIND_SIMILAR_COMPANIES_SQL = """
SELECT
    c.registration_number,
    c.company_name,
    c.industry,
    c.state,
    (1.0 - (c.embedding <=> src.embedding))::double precision AS similarity
FROM companies AS c
CROSS JOIN (
    SELECT embedding, state, industry
    FROM companies
    WHERE registration_number = %s
) AS src
WHERE
    c.registration_number <> %s
    AND c.status = 'active'
    AND c.embedding IS NOT NULL
    AND src.embedding IS NOT NULL
ORDER BY
    ROUND((c.embedding <=> src.embedding)::numeric, 5) ASC,
    CASE WHEN src.state IS NOT NULL AND c.state = src.state THEN 1 ELSE 0 END DESC,
    CASE WHEN src.industry IS NOT NULL AND c.industry = src.industry THEN 1 ELSE 0 END DESC,
    c.registration_number ASC
LIMIT %s
"""

_FIND_INDUSTRY_FALLBACK_SQL = """
SELECT
    registration_number,
    company_name,
    industry,
    state
FROM companies
WHERE
    registration_number <> %s
    AND status = 'active'
    AND industry = %s
ORDER BY
    CASE WHEN %s::text IS NOT NULL AND state = %s::text THEN 1 ELSE 0 END DESC,
    CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END DESC,
    registration_number ASC
LIMIT %s
"""


def _configure_connection(conn: psycopg.Connection[Any]) -> None:
    """Register pgvector adapter on pooled connections without crashing before migration."""
    try:
        register_vector(conn)
    except ValueError:
        logger.warning("pgvector extension not yet installed; vector adapter skipped")


def apply_migrations(
    database_url: str,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list[str]:
    """Apply all SQL migration files in lexical order within a transaction."""
    sql_files = sorted(migrations_dir.glob("*.sql"))
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            for sql_file in sql_files:
                cur.execute(sql_file.read_text(encoding="utf-8"))
                applied.append(sql_file.name)
        conn.commit()
    return applied


def create_pool(
    database_url: str,
    *,
    min_size: int = 1,
    max_size: int = 10,
) -> ConnectionPool[Any]:
    """Create a resilient psycopg pool that validates idle connections and registers pgvector."""
    return ConnectionPool(
        conninfo=database_url,
        min_size=min_size,
        max_size=max_size,
        configure=_configure_connection,
        check=ConnectionPool.check_connection,
        open=True,
    )


def _vector_to_list(raw_embedding: Any) -> list[float] | None:
    if raw_embedding is None:
        return None
    if isinstance(raw_embedding, Vector):
        return [float(x) for x in raw_embedding.to_list()]
    return [float(x) for x in raw_embedding]


def get_company_version(
    conn: psycopg.Connection[Any],
    registration_number: str,
) -> StoredCompanyVersion | None:
    """Fetch stored version and embedding metadata for preliminary optimization checks."""
    with conn.cursor() as cur:
        cur.execute(_GET_COMPANY_VERSION_SQL, (registration_number,))
        row = cur.fetchone()
    if row is None:
        return None
    embedding = _vector_to_list(row[5])
    return StoredCompanyVersion(
        registration_number=row[0],
        status=CompanyStatus(row[1]),
        updated_at=row[2],
        event_version_key=row[3],
        has_metadata=bool(row[4]),
        embedding=embedding,
        embedding_model=row[6],
        embedding_text_hash=row[7],
    )


def upsert_company(
    conn: psycopg.Connection[Any],
    event: UpsertCompanyEvent,
    *,
    embedding: list[float] | None,
    embedding_model: str | None,
    embedding_text_hash: str | None,
) -> bool:
    """Atomically upsert company data and embedding if incoming source version wins."""
    vector_param = Vector(embedding) if embedding is not None else None
    with conn.cursor() as cur:
        cur.execute(
            _UPSERT_COMPANY_SQL,
            (
                event.registration_number,
                event.company_name,
                event.industry,
                event.state,
                event.nature_of_business,
                event.updated_at,
                event.raw_updated_at,
                event.event_version_key,
                event.message_id,
                vector_param,
                embedding_model,
                embedding_text_hash,
            ),
        )
        return cur.fetchone() is not None


def mark_company_struck_off(
    conn: psycopg.Connection[Any],
    event: StruckOffCompanyEvent,
) -> bool:
    """Mark a company as struck_off if incoming source version wins, preserving metadata."""
    with conn.cursor() as cur:
        cur.execute(
            _STRUCK_OFF_COMPANY_SQL,
            (
                event.registration_number,
                event.updated_at,
                event.raw_updated_at,
                event.event_version_key,
                event.message_id,
            ),
        )
        return cur.fetchone() is not None


def record_ingestion_error(
    conn: psycopg.Connection[Any],
    error: QuarantineRecord,
) -> None:
    """Durably quarantine a malformed or unprocessable event."""
    with conn.cursor() as cur:
        cur.execute(
            _RECORD_INGESTION_ERROR_SQL,
            (
                error.message_id,
                error.registration_number,
                error.error_type,
                error.error_message,
                error.raw_payload,
            ),
        )


def get_company_profile(
    conn: psycopg.Connection[Any],
    registration_number: str,
) -> CompanyProfile | None:
    """Fetch stored portal metadata and embedding presence for one company."""
    with conn.cursor() as cur:
        cur.execute(_GET_COMPANY_PROFILE_SQL, (registration_number,))
        row = cur.fetchone()
    if row is None:
        return None
    return CompanyProfile(
        registration_number=row[0],
        company_name=row[1],
        industry=row[2],
        state=row[3],
        nature_of_business=row[4],
        status=CompanyStatus(row[5]),
        has_embedding=bool(row[6]),
    )


def find_similar_companies(
    conn: psycopg.Connection[Any],
    registration_number: str,
    *,
    limit: int,
) -> list[SimilarCompanyRow]:
    """Run exact pgvector cosine similarity search in PostgreSQL, excluding the source company."""
    with conn.cursor() as cur:
        cur.execute(
            _FIND_SIMILAR_COMPANIES_SQL,
            (registration_number, registration_number, limit),
        )
        rows = cur.fetchall()

    results: list[SimilarCompanyRow] = []
    for row in rows:
        score = max(-1.0, min(1.0, float(row[4])))
        results.append(
            SimilarCompanyRow(
                registration_number=row[0],
                company_name=row[1],
                industry=row[2],
                state=row[3],
                similarity=round(score, 4),
                match_type=MatchType.SEMANTIC,
            )
        )
    return results


def find_industry_fallback_companies(
    conn: psycopg.Connection[Any],
    registration_number: str,
    *,
    industry: str,
    state: str | None,
    limit: int,
) -> list[SimilarCompanyRow]:
    """Lower-confidence metadata fallback when the source company lacks an embedding."""
    with conn.cursor() as cur:
        cur.execute(
            _FIND_INDUSTRY_FALLBACK_SQL,
            (registration_number, industry, state, state, limit),
        )
        rows = cur.fetchall()

    return [
        SimilarCompanyRow(
            registration_number=row[0],
            company_name=row[1],
            industry=row[2],
            state=row[3],
            similarity=None,
            match_type=MatchType.INDUSTRY_FALLBACK,
        )
        for row in rows
    ]


def check_database_health(conn: psycopg.Connection[Any]) -> bool:
    """Verify database connectivity with a minimal query."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        return cur.fetchone() is not None


def main() -> None:
    settings = Settings()
    applied = apply_migrations(settings.effective_database_url)
    print(f"Applied migrations: {', '.join(applied)}")


if __name__ == "__main__":
    main()
