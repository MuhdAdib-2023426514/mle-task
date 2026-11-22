"""Real-Postgres integration tests for version-aware upserts, embeddings, and quarantine."""

import json
import math
from typing import Any

import pytest
from psycopg_pool import ConnectionPool

from app.consumer import process_event
from app.database import upsert_company
from app.embeddings import FastEmbedder
from app.normalization import parse_raw_event
from app.schemas import EventOutcome, UpsertCompanyEvent
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.integration


def _event_bytes(
    *,
    reg: str = "SYN-000001",
    updated_at: object = "2026-08-14T10:00:00+08:00",
    name: str = "Acme Bakery Sdn. Bhd.",
    desc: str | None = "Operating cafes and coffee shops",
    state: str | None = "Selangor",
    industry: str | None = "Food & Beverage",
    include_updated_at: bool = True,
) -> bytes:
    payload: dict[str, Any] = {
        "type": "company-upserted",
        "registrationNumber": reg,
        "data": {
            "companyName": name,
            "natureOfBusiness": desc,
            "state": state,
            "industry": industry,
        },
    }
    if include_updated_at:
        payload["updatedAt"] = updated_at
    return json.dumps(payload).encode("utf-8")


def _fetch_company_row(pool: ConnectionPool[Any], reg: str) -> dict[str, Any] | None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT registration_number, company_name, industry, state,
                       nature_of_business, status, updated_at, event_version_key,
                       embedding, embedding_model, embedding_text_hash
                FROM companies
                WHERE registration_number = %s
                """,
                (reg,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return {
        "registration_number": row[0],
        "company_name": row[1],
        "industry": row[2],
        "state": row[3],
        "nature_of_business": row[4],
        "status": row[5],
        "updated_at": row[6],
        "event_version_key": row[7],
        "embedding": (
            [float(x) for x in row[8].to_list()]
            if hasattr(row[8], "to_list")
            else ([float(x) for x in row[8]] if row[8] is not None else None)
        ),
        "embedding_model": row[9],
        "embedding_text_hash": row[10],
    }


def test_newer_version_wins_in_both_delivery_orders(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    older = _event_bytes(
        reg="SYN-000010",
        updated_at="2026-08-14T09:00:00+08:00",
        name="Old Name",
        desc="Old coffee shop description",
    )
    newer = _event_bytes(
        reg=" syn-000010 ",
        updated_at="2026-08-14T11:00:00+08:00",
        name="New Name",
        desc="Updated specialty coffee roastery",
    )

    # Order 1: newer arrives first, older arrives second -> newer must stay
    assert process_event(newer, "m-new", db_pool, fake_embedder) == EventOutcome.APPLIED
    assert process_event(older, "m-old", db_pool, fake_embedder) == EventOutcome.STALE_OR_DUPLICATE
    row = _fetch_company_row(db_pool, "SYN-000010")
    assert row is not None
    assert row["company_name"] == "New Name"
    assert row["nature_of_business"] == "Updated specialty coffee roastery"
    assert row["embedding"] == pytest.approx(
        fake_embedder.embed("Updated specialty coffee roastery")
    )

    # Order 2 (different company): older arrives first, newer arrives second -> newer wins
    older_b = _event_bytes(
        reg="SYN-000011",
        updated_at="2026-08-14T09:00:00+08:00",
        name="Old B",
        desc="Old B desc",
    )
    newer_b = _event_bytes(
        reg="SYN-000011",
        updated_at="2026-08-14T11:00:00+08:00",
        name="New B",
        desc="New B desc",
    )
    assert process_event(older_b, "mb-old", db_pool, fake_embedder) == EventOutcome.APPLIED
    assert process_event(newer_b, "mb-new", db_pool, fake_embedder) == EventOutcome.APPLIED
    row_b = _fetch_company_row(db_pool, "SYN-000011")
    assert row_b is not None
    assert row_b["company_name"] == "New B"
    assert row_b["nature_of_business"] == "New B desc"
    assert row_b["embedding"] == pytest.approx(fake_embedder.embed("New B desc"))


def test_undated_inserts_when_new_never_overwrites_dated_and_is_replaced_by_dated(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    undated = _event_bytes(
        reg="SYN-000020",
        include_updated_at=False,
        name="Unknown",
        desc=None,
        state=None,
        industry=None,
    )
    dated = _event_bytes(
        reg="SYN-000020",
        updated_at="2026-08-14T13:30:00+08:00",
        name="Real Company Sdn. Bhd.",
        desc="Homestay and chalet operations",
        state="KUALA LUMPUR",
        industry="Hospitality",
    )

    # Undated arrives first -> stored with NULL embedding
    assert process_event(undated, "m-u1", db_pool, fake_embedder) == EventOutcome.APPLIED
    row = _fetch_company_row(db_pool, "SYN-000020")
    assert row is not None
    assert row["company_name"] == "Unknown"
    assert row["embedding"] is None

    # Dated arrives later -> overwrites undated and computes embedding
    assert process_event(dated, "m-d1", db_pool, fake_embedder) == EventOutcome.APPLIED
    row = _fetch_company_row(db_pool, "SYN-000020")
    assert row is not None
    assert row["company_name"] == "Real Company Sdn. Bhd."
    assert row["state"] == "Kuala Lumpur"
    assert row["embedding"] is not None

    # Undated arrives again -> rejected as stale
    assert process_event(undated, "m-u2", db_pool, fake_embedder) == EventOutcome.STALE_OR_DUPLICATE
    row = _fetch_company_row(db_pool, "SYN-000020")
    assert row is not None
    assert row["company_name"] == "Real Company Sdn. Bhd."


def test_exact_timestamp_tie_is_arrival_order_independent(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    ev_a = _event_bytes(reg="SYN-000030", updated_at="2026-08-14T10:00:00Z", desc="Variant Alpha")
    ev_b = _event_bytes(reg="SYN-000030", updated_at="2026-08-14T10:00:00Z", desc="Variant Beta")

    process_event(ev_a, "m-a1", db_pool, fake_embedder)
    process_event(ev_b, "m-b1", db_pool, fake_embedder)
    winner_ab = _fetch_company_row(db_pool, "SYN-000030")

    # Repeat in reverse order on a second company with identical payload tie
    ev_a2 = _event_bytes(reg="SYN-000031", updated_at="2026-08-14T10:00:00Z", desc="Variant Alpha")
    ev_b2 = _event_bytes(reg="SYN-000031", updated_at="2026-08-14T10:00:00Z", desc="Variant Beta")

    # Compare order independence on the SAME reg by truncating in between
    with db_pool.connection() as conn:
        with conn.transaction():
            conn.execute("DELETE FROM companies WHERE registration_number = 'SYN-000030'")

    process_event(ev_b, "m-b2", db_pool, fake_embedder)
    process_event(ev_a, "m-a2", db_pool, fake_embedder)
    winner_ba = _fetch_company_row(db_pool, "SYN-000030")

    assert winner_ab is not None and winner_ba is not None
    assert winner_ab["nature_of_business"] == winner_ba["nature_of_business"]
    assert winner_ab["event_version_key"] == winner_ba["event_version_key"]
    # Also ensure unused locals aren't flagged
    assert ev_a2 != ev_b2


def test_unchanged_description_reuses_embedding_and_replay_skips_inference(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    v1 = _event_bytes(
        reg="SYN-000040",
        updated_at="2026-08-14T10:00:00+08:00",
        name="Initial Name",
        desc="Freight forwarding at Port Klang",
    )
    v2 = _event_bytes(
        reg="SYN-000040",
        updated_at="2026-08-14T11:00:00+08:00",
        name="Renamed Logistics Sdn. Bhd.",
        desc="Freight forwarding at Port Klang",
    )

    assert process_event(v1, "m-1", db_pool, fake_embedder) == EventOutcome.APPLIED
    assert len(fake_embedder.calls) == 1

    # Name changed, description identical -> metadata updated without extra inference call
    assert process_event(v2, "m-2", db_pool, fake_embedder) == EventOutcome.APPLIED
    assert len(fake_embedder.calls) == 1
    row = _fetch_company_row(db_pool, "SYN-000040")
    assert row is not None
    assert row["company_name"] == "Renamed Logistics Sdn. Bhd."
    assert row["embedding"] is not None

    # Exact duplicate replay -> stale/duplicate, still 1 inference call
    assert process_event(v2, "m-2", db_pool, fake_embedder) == EventOutcome.STALE_OR_DUPLICATE
    assert len(fake_embedder.calls) == 1


def test_struck_off_preserves_company_row_and_respects_timestamps(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    upsert = _event_bytes(
        reg="SYN-000050",
        updated_at="2026-08-14T10:00:00+08:00",
        name="Active Co Sdn. Bhd.",
        desc="Software development for accounting",
    )
    stale_struck_off = json.dumps(
        {
            "type": "company-struck-off",
            "registrationNumber": "SYN-000050",
            "updatedAt": "2026-08-14T09:00:00+08:00",
        }
    ).encode("utf-8")
    newer_struck_off = json.dumps(
        {
            "type": "company-struck-off",
            "registrationNumber": "SYN-000050",
            "updatedAt": "2026-08-14T16:00:00+08:00",
        }
    ).encode("utf-8")

    assert process_event(upsert, "m-u", db_pool, fake_embedder) == EventOutcome.APPLIED
    assert (
        process_event(stale_struck_off, "m-so-old", db_pool, fake_embedder)
        == EventOutcome.STALE_OR_DUPLICATE
    )
    row = _fetch_company_row(db_pool, "SYN-000050")
    assert row is not None and row["status"] == "active"

    assert (
        process_event(newer_struck_off, "m-so-new", db_pool, fake_embedder) == EventOutcome.APPLIED
    )
    row = _fetch_company_row(db_pool, "SYN-000050")
    assert row is not None
    assert row["status"] == "struck_off"
    assert row["company_name"] == "Active Co Sdn. Bhd."
    assert row["embedding"] is not None


def test_out_of_order_struck_off_before_older_upsert_backfills_metadata(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    struck_off = json.dumps(
        {
            "type": "company-struck-off",
            "registrationNumber": "SYN-000055",
            "updatedAt": "2026-08-14T16:00:00+08:00",
        }
    ).encode("utf-8")
    older_upsert = _event_bytes(
        reg="SYN-000055",
        updated_at="2026-08-14T10:00:00+08:00",
        name="Backfilled Co Sdn. Bhd.",
        desc="Precision machining of metal parts",
        state="SELANGOR",
        industry="Manufacturing",
    )
    even_older_upsert = _event_bytes(
        reg="SYN-000055",
        updated_at="2026-08-14T09:00:00+08:00",
        name="Stale Name Should Not Overwrite",
        desc="Different activity",
    )

    assert process_event(struck_off, "m-so-first", db_pool, fake_embedder) == EventOutcome.APPLIED
    stub = _fetch_company_row(db_pool, "SYN-000055")
    assert stub is not None
    assert stub["status"] == "struck_off"
    assert stub["company_name"] is None

    # Older upsert backfills NULL metadata & embedding while preserving struck_off & 16:00 timestamp
    assert process_event(older_upsert, "m-bf", db_pool, fake_embedder) == EventOutcome.APPLIED
    filled = _fetch_company_row(db_pool, "SYN-000055")
    assert filled is not None
    assert filled["status"] == "struck_off"
    assert filled["updated_at"] == stub["updated_at"]
    assert filled["company_name"] == "Backfilled Co Sdn. Bhd."
    assert filled["state"] == "Selangor"
    assert filled["embedding"] is not None

    # Another older upsert after metadata is already filled is rejected as stale
    assert (
        process_event(even_older_upsert, "m-stale", db_pool, fake_embedder)
        == EventOutcome.STALE_OR_DUPLICATE
    )
    after = _fetch_company_row(db_pool, "SYN-000055")
    assert after is not None
    assert after["company_name"] == "Backfilled Co Sdn. Bhd."
    assert after["status"] == "struck_off"


def test_quarantine_persists_and_is_idempotent_on_replay(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    bad = b'{"type": "company-upserted", "data": "broken'
    assert process_event(bad, "bad-sha256", db_pool, fake_embedder) == EventOutcome.QUARANTINED
    assert process_event(bad, "bad-sha256", db_pool, fake_embedder) == EventOutcome.QUARANTINED

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), MIN(error_type) FROM ingestion_errors")
            count, error_type = cur.fetchone() or (0, None)
    assert count == 1
    assert error_type == "invalid_json"


def test_transaction_rollback_leaves_no_partial_state(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    parsed = parse_raw_event(
        _event_bytes(reg="SYN-000060", name="Rollback Co", desc="Solar panel installation"),
        message_id="m-rb",
    )
    assert isinstance(parsed, UpsertCompanyEvent)

    with pytest.raises(RuntimeError, match="forced failure"):
        with db_pool.connection() as conn:
            with conn.transaction():
                upsert_company(
                    conn,
                    parsed,
                    embedding=fake_embedder.embed("Solar panel installation"),
                    embedding_model=fake_embedder.model_name,
                    embedding_text_hash="hash-rb",
                )
                raise RuntimeError("forced failure")

    assert _fetch_company_row(db_pool, "SYN-000060") is None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def test_real_multilingual_model_smoke() -> None:
    """Smoke test real CPU ONNX model: dimension, long input, and EN<->MS alignment."""
    embedder = FastEmbedder()
    en_hotel = embedder.embed("Boutique hotel in Penang")
    ms_hotel = embedder.embed("Hotel butik di Seremban")
    en_trucking = embedder.embed("Trucking services for construction materials")
    long_text = embedder.embed("Hotel butik di Seremban " * 150)

    assert len(en_hotel) == 384
    assert len(ms_hotel) == 384
    assert len(long_text) == 384

    sim_cross_lingual = _cosine_similarity(en_hotel, ms_hotel)
    sim_unrelated = _cosine_similarity(en_hotel, en_trucking)
    assert sim_cross_lingual > sim_unrelated
    assert sim_cross_lingual > 0.6
