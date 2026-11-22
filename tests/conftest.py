"""Shared test fixtures: deterministic FakeEmbedder and isolated Postgres test DB."""

import hashlib
import math
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.database import apply_migrations, create_pool

TEST_DB_NAME = "companies_test"


class FakeEmbedder:
    """Deterministic 384-dim unit-vector embedder for fast, repeatable tests."""

    def __init__(self, model_name: str = "fake-embedder-v1", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[i % len(digest)] + i) % 251) - 125.0 for i in range(self._dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def _ensure_test_database(settings: Settings) -> str:
    admin_settings = settings.model_copy(update={"postgres_db": "postgres", "database_url": None})
    test_settings = settings.model_copy(update={"postgres_db": TEST_DB_NAME, "database_url": None})

    with psycopg.connect(admin_settings.effective_database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')

    return test_settings.effective_database_url


@pytest.fixture(scope="session")
def test_db_url() -> str:
    settings = Settings()
    url = _ensure_test_database(settings)
    apply_migrations(url)
    return url


@pytest.fixture(scope="session")
def _session_db_pool(test_db_url: str) -> Iterator[ConnectionPool[Any]]:
    pool = create_pool(test_db_url, min_size=1, max_size=4)
    try:
        yield pool
    finally:
        pool.close()


@pytest.fixture
def db_pool(_session_db_pool: ConnectionPool[Any]) -> ConnectionPool[Any]:
    with _session_db_pool.connection() as conn:
        with conn.transaction():
            conn.execute("TRUNCATE TABLE companies, ingestion_errors RESTART IDENTITY")
    return _session_db_pool
