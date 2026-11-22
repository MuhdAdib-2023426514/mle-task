"""Consumer reliability tests: commit-before-ACK, failure handling, and RabbitMQ redelivery."""

import json
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import aio_pika
import psycopg
import pytest
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.consumer import handle_message
from app.schemas import EventOutcome
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.integration


class _FakeIncomingMessage:
    """Minimal AMQP message double recording ACK/NACK order and arguments."""

    def __init__(
        self,
        body: bytes,
        message_id: str | None = "msg-1",
        on_ack: Callable[[], None] | None = None,
    ) -> None:
        self.body = body
        self.message_id = message_id
        self._on_ack = on_ack
        self.acked = False
        self.nacked = False
        self.requeued: bool | None = None

    async def ack(self, multiple: bool = False) -> None:
        if self._on_ack is not None:
            self._on_ack()
        self.acked = True

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None:
        self.nacked = True
        self.requeued = requeue


class _FailingPool:
    """Pool double that raises OperationalError on every connection attempt."""

    def __init__(self) -> None:
        self.attempts = 0

    @contextmanager
    def connection(self) -> Any:
        self.attempts += 1
        raise psycopg.OperationalError("simulated database outage")
        yield  # pragma: no cover


def _valid_event_bytes(reg: str = "SYN-000701") -> bytes:
    return json.dumps(
        {
            "type": "company-upserted",
            "registrationNumber": reg,
            "updatedAt": "2026-08-14T10:00:00+08:00",
            "data": {
                "companyName": "Reliable Co Sdn. Bhd.",
                "natureOfBusiness": "Cold chain logistics and warehousing",
                "state": "Selangor",
                "industry": "Logistics",
            },
        }
    ).encode("utf-8")


async def test_handle_message_acks_only_after_db_commit(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    row_visible_at_ack = False

    def verify_committed_before_ack() -> None:
        nonlocal row_visible_at_ack
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT company_name FROM companies WHERE registration_number = %s",
                    ("SYN-000701",),
                )
                row = cur.fetchone()
                row_visible_at_ack = row is not None and row[0] == "Reliable Co Sdn. Bhd."

    msg = _FakeIncomingMessage(
        body=_valid_event_bytes("SYN-000701"),
        message_id="msg-commit-first",
        on_ack=verify_committed_before_ack,
    )
    settings = Settings(max_transient_retries=1, retry_backoff_seconds=0.01)

    outcome = await handle_message(msg, db_pool, fake_embedder, settings)  # type: ignore[arg-type]

    assert outcome == EventOutcome.APPLIED
    assert msg.acked is True
    assert msg.nacked is False
    assert row_visible_at_ack is True


async def test_handle_message_does_not_ack_on_db_commit_failure(
    fake_embedder: FakeEmbedder,
) -> None:
    failing_pool = _FailingPool()
    msg = _FakeIncomingMessage(body=_valid_event_bytes("SYN-000702"), message_id="msg-db-down")
    settings = Settings(max_transient_retries=2, retry_backoff_seconds=0.01)

    outcome = await handle_message(
        msg,  # type: ignore[arg-type]
        failing_pool,  # type: ignore[arg-type]
        fake_embedder,
        settings,
    )

    assert outcome is None
    assert failing_pool.attempts == 3  # 1 initial + 2 retries
    assert msg.acked is False
    assert msg.nacked is True
    assert msg.requeued is True


async def test_handle_message_quarantines_bad_payload_before_ack(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    quarantine_visible_at_ack = False

    def verify_quarantined_before_ack() -> None:
        nonlocal quarantine_visible_at_ack
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT error_type FROM ingestion_errors WHERE message_id = %s",
                    ("msg-bad-json",),
                )
                row = cur.fetchone()
                quarantine_visible_at_ack = row is not None and row[0] == "invalid_json"

    msg = _FakeIncomingMessage(
        body=b'{"type": "company-upserted", "data": "truncated',
        message_id="msg-bad-json",
        on_ack=verify_quarantined_before_ack,
    )
    settings = Settings(max_transient_retries=1, retry_backoff_seconds=0.01)

    outcome = await handle_message(msg, db_pool, fake_embedder, settings)  # type: ignore[arg-type]

    assert outcome == EventOutcome.QUARANTINED
    assert msg.acked is True
    assert msg.nacked is False
    assert quarantine_visible_at_ack is True


async def test_handle_message_does_not_ack_quarantine_when_db_down(
    fake_embedder: FakeEmbedder,
) -> None:
    failing_pool = _FailingPool()
    msg = _FakeIncomingMessage(body=b"not-json", message_id="msg-bad-db-down")
    settings = Settings(max_transient_retries=1, retry_backoff_seconds=0.01)

    outcome = await handle_message(
        msg,  # type: ignore[arg-type]
        failing_pool,  # type: ignore[arg-type]
        fake_embedder,
        settings,
    )

    assert outcome is None
    assert msg.acked is False
    assert msg.nacked is True
    assert msg.requeued is True


class _ExplodingEmbedder:
    """Embedder double raising a deterministic ValueError to simulate a poison-pill bug."""

    model_name = "broken-embedder"
    dimension = 384

    def embed(self, text: str) -> list[float]:
        raise ValueError("deterministic vector error")


async def test_handle_message_quarantines_deterministic_processing_error_before_ack(
    db_pool: ConnectionPool[Any],
) -> None:
    quarantine_visible_at_ack = False

    def verify_quarantined_before_ack() -> None:
        nonlocal quarantine_visible_at_ack
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT error_type, error_message FROM ingestion_errors WHERE message_id = %s",
                    ("msg-poison-pill",),
                )
                row = cur.fetchone()
                quarantine_visible_at_ack = (
                    row is not None
                    and row[0] == "processing_error"
                    and "deterministic vector error" in row[1]
                )

    msg = _FakeIncomingMessage(
        body=_valid_event_bytes("SYN-000705"),
        message_id="msg-poison-pill",
        on_ack=verify_quarantined_before_ack,
    )
    settings = Settings(max_transient_retries=1, retry_backoff_seconds=0.01)

    outcome = await handle_message(
        msg,  # type: ignore[arg-type]
        db_pool,
        _ExplodingEmbedder(),
        settings,
    )

    assert outcome == EventOutcome.QUARANTINED
    assert msg.acked is True
    assert msg.nacked is False
    assert quarantine_visible_at_ack is True


def test_pool_check_connection_recovers_after_closed_connection(
    db_pool: ConnectionPool[Any],
) -> None:
    # Terminate a pooled connection from another session and ensure the pool heals transparently
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            pid = cur.fetchone()[0]  # type: ignore[index]

    with db_pool.connection() as killer:
        with killer.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(%s)", (pid,))

    with db_pool.connection() as recovered:
        with recovered.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)


async def test_rabbitmq_nack_redelivery_and_eventual_commit(
    db_pool: ConnectionPool[Any],
    fake_embedder: FakeEmbedder,
) -> None:
    """Verify real RabbitMQ broker redelivers after DB failure and commits cleanly on recovery."""
    settings = Settings(max_transient_retries=0, retry_backoff_seconds=0.01)
    temp_queue_name = f"test-company-events-{uuid.uuid4().hex[:10]}"
    payload = _valid_event_bytes("SYN-000799")

    connection = await aio_pika.connect_robust(settings.effective_rabbitmq_url)
    try:
        async with connection:
            channel = await connection.channel(publisher_confirms=True)
            queue = await channel.declare_queue(temp_queue_name, durable=True)
            try:
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=payload,
                        message_id="rmq-redelivery-test-1",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=temp_queue_name,
                )

                # First delivery: simulate DB failure -> nack(requeue=True), row not persisted
                first_msg = await queue.get(timeout=5.0)
                assert first_msg is not None
                failing_pool = _FailingPool()
                first_outcome = await handle_message(
                    first_msg,
                    failing_pool,  # type: ignore[arg-type]
                    fake_embedder,
                    settings,
                )
                assert first_outcome is None

                with db_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT COUNT(*) FROM companies WHERE registration_number = %s",
                            ("SYN-000799",),
                        )
                        assert cur.fetchone() == (0,)

                # Second delivery: broker redelivers -> healthy DB commits and ACKs
                second_msg = await queue.get(timeout=5.0)
                assert second_msg is not None
                assert second_msg.redelivered is True
                second_outcome = await handle_message(second_msg, db_pool, fake_embedder, settings)
                assert second_outcome == EventOutcome.APPLIED

                with db_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT company_name, embedding IS NOT NULL "
                            "FROM companies WHERE registration_number = %s",
                            ("SYN-000799",),
                        )
                        assert cur.fetchone() == ("Reliable Co Sdn. Bhd.", True)

                # Queue is now empty (no unacked or lingering message)
                assert await queue.get(fail=False) is None
            finally:
                await queue.delete(if_unused=False, if_empty=False)
    finally:
        pass
