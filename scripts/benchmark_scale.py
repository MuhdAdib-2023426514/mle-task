"""Reproducible Tier 3 scale benchmark: exact scan vs. pgvector HNSW & ONNX batching.

Runs in the isolated `companies_test` database using a temporary `bench_vectors`
table (dropped on exit) so the primary `companies` database is never modified.
"""

import time

import numpy as np
import psycopg
from fastembed import TextEmbedding
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.config import Settings

TEST_DB_NAME = "companies_test"
SCALES = [2_000, 20_000, 50_000]
DIM = 384
NUM_QUERIES = 50
TOP_K = 5


def _ensure_test_db(settings: Settings) -> str:
    admin_url = settings.model_copy(
        update={"postgres_db": "postgres", "database_url": None}
    ).effective_database_url
    test_url = settings.model_copy(
        update={"postgres_db": TEST_DB_NAME, "database_url": None}
    ).effective_database_url
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    return test_url


def _generate_clustered_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate realistic clustered unit vectors around 50 topic centroids."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(50, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    assignments = rng.integers(0, 50, size=n)
    noise = rng.normal(scale=0.18, size=(n, dim)).astype(np.float32)
    vecs = centroids[assignments] + noise
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def _run_queries(
    conn: psycopg.Connection[object],
    query_vecs: list[Vector],
    *,
    force_seq_scan: bool,
) -> tuple[list[float], list[list[int]]]:
    latencies_ms: list[float] = []
    results: list[list[int]] = []
    with conn.cursor() as cur:
        cur.execute(f"SET enable_indexscan = {'off' if force_seq_scan else 'on'}")
        cur.execute(f"SET enable_bitmapscan = {'off' if force_seq_scan else 'on'}")
        for qv in query_vecs:
            t0 = time.perf_counter()
            cur.execute(
                """
                SELECT id
                FROM bench_vectors
                ORDER BY embedding <=> %s ASC
                LIMIT %s
                """,
                (qv, TOP_K),
            )
            ids = [row[0] for row in cur.fetchall()]
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            results.append(ids)
    return latencies_ms, results


def benchmark_pgvector(test_url: str) -> None:
    print("=== PostgreSQL pgvector Scale Benchmark (Exact Cosine vs. HNSW) ===")
    print(f"Queries per scale: {NUM_QUERIES}, top-k: {TOP_K}, dim: {DIM}")

    with psycopg.connect(test_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        try:
            for n in SCALES:
                vecs = _generate_clustered_vectors(n, DIM, seed=42 + n)
                query_vecs = [Vector(vecs[i].tolist()) for i in range(NUM_QUERIES)]

                with conn.cursor() as cur:
                    cur.execute("DROP TABLE IF EXISTS bench_vectors")
                    cur.execute(
                        f"CREATE TABLE bench_vectors (id INT PRIMARY KEY, embedding vector({DIM}))"
                    )
                    with cur.copy("COPY bench_vectors (id, embedding) FROM STDIN") as copy:
                        for idx in range(n):
                            vec_str = "[" + ",".join(f"{x:.6f}" for x in vecs[idx].tolist()) + "]"
                            copy.write_row((idx, vec_str))
                    cur.execute("ANALYZE bench_vectors")

                # Warm up cache
                _run_queries(conn, query_vecs[:5], force_seq_scan=True)
                exact_ms, exact_ids = _run_queries(conn, query_vecs, force_seq_scan=True)

                t_idx = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(
                        "CREATE INDEX idx_bench_hnsw ON bench_vectors "
                        "USING hnsw (embedding vector_cosine_ops) "
                        "WITH (m = 16, ef_construction = 64)"
                    )
                hnsw_build_s = time.perf_counter() - t_idx

                _run_queries(conn, query_vecs[:5], force_seq_scan=False)
                hnsw_ms, hnsw_ids = _run_queries(conn, query_vecs, force_seq_scan=False)

                recall_scores = [
                    len(set(ex) & set(ap)) / TOP_K
                    for ex, ap in zip(exact_ids, hnsw_ids, strict=True)
                ]
                print(
                    f"N={n:6d} | "
                    f"Exact p50={np.percentile(exact_ms, 50):5.2f}ms "
                    f"p95={np.percentile(exact_ms, 95):5.2f}ms | "
                    f"HNSW build={hnsw_build_s:5.2f}s "
                    f"p50={np.percentile(hnsw_ms, 50):5.2f}ms "
                    f"p95={np.percentile(hnsw_ms, 95):5.2f}ms "
                    f"Recall@{TOP_K}={np.mean(recall_scores):.4f}"
                )
        finally:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS bench_vectors")


def benchmark_onnx_batching(primary_url: str, model_name: str) -> None:
    print(f"\n=== ONNX CPU Micro-Batch Throughput ({model_name}) ===")
    with psycopg.connect(primary_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT nature_of_business
                FROM companies
                WHERE nature_of_business IS NOT NULL
                ORDER BY registration_number
                LIMIT 256
                """
            )
            texts = [row[0] for row in cur.fetchall()]

    model = TextEmbedding(model_name=model_name)
    # Warm-up
    list(model.embed(texts[:4]))

    for batch_size in (1, 16, 64):
        t0 = time.perf_counter()
        if batch_size == 1:
            for t in texts:
                next(iter(model.embed([t])))
        else:
            list(model.embed(texts, batch_size=batch_size))
        elapsed = time.perf_counter() - t0
        print(
            f"batch_size={batch_size:2d} | {len(texts)} docs in {elapsed:.3f}s "
            f"({len(texts) / elapsed:6.1f} docs/sec)"
        )


def main() -> None:
    settings = Settings()
    test_url = _ensure_test_db(settings)
    benchmark_pgvector(test_url)
    benchmark_onnx_batching(settings.effective_database_url, settings.embedding_model)


if __name__ == "__main__":
    main()
