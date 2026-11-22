"""RabbitMQ consumer for company-events with commit-before-ACK semantics."""

import asyncio
import logging
import signal
from typing import Any

import aio_pika
import psycopg
from aio_pika.abc import AbstractIncomingMessage
from psycopg_pool import ConnectionPool, PoolTimeout

from app.config import Settings
from app.database import (
    apply_migrations,
    create_pool,
    get_company_version,
    mark_company_struck_off,
    record_ingestion_error,
    upsert_company,
)
from app.embeddings import Embedder, FastEmbedder
from app.normalization import (
    compute_embedding_text_hash,
    incoming_version_wins,
    parse_raw_event,
    prepare_embedding_text,
)
from app.schemas import (
    CompanyStatus,
    EventOutcome,
    QuarantineRecord,
    StruckOffCompanyEvent,
    UpsertCompanyEvent,
)

logger = logging.getLogger(__name__)


def process_event(
    raw_body: bytes,
    message_id: str | None,
    pool: ConnectionPool[Any],
    embedder: Embedder,
) -> EventOutcome:
    """Parse, embed (when needed), and durably commit one company event.

    All database mutations commit before returning so callers can safely ACK
    only after durable persistence succeeds.
    """
    parsed = parse_raw_event(raw_body, message_id)

    if isinstance(parsed, QuarantineRecord):
        with pool.connection() as conn:
            with conn.transaction():
                record_ingestion_error(conn, parsed)
        logger.warning(
            "Quarantined event message_id=%s reg=%s error_type=%s",
            parsed.message_id,
            parsed.registration_number,
            parsed.error_type,
        )
        return EventOutcome.QUARANTINED

    if isinstance(parsed, StruckOffCompanyEvent):
        with pool.connection() as conn:
            with conn.transaction():
                applied = mark_company_struck_off(conn, parsed)
        return EventOutcome.APPLIED if applied else EventOutcome.STALE_OR_DUPLICATE

    return _process_upsert_event(parsed, pool, embedder)


def _process_upsert_event(
    event: UpsertCompanyEvent,
    pool: ConnectionPool[Any],
    embedder: Embedder,
) -> EventOutcome:
    prepared_text = prepare_embedding_text(event.nature_of_business)
    text_hash = (
        compute_embedding_text_hash(embedder.model_name, prepared_text)
        if prepared_text is not None
        else None
    )

    # Preliminary freshness/cache read is strictly an optimization to avoid wasted
    # CPU inference; the database-side conditional upsert enforces final correctness.
    with pool.connection() as conn:
        existing = get_company_version(conn, event.registration_number)

    needs_struck_off_backfill = (
        existing is not None
        and existing.status == CompanyStatus.STRUCK_OFF
        and not existing.has_metadata
        and event.company_name is not None
    )

    if (
        existing is not None
        and not needs_struck_off_backfill
        and not incoming_version_wins(
            existing_updated_at=existing.updated_at,
            existing_version_key=existing.event_version_key,
            incoming_updated_at=event.updated_at,
            incoming_version_key=event.event_version_key,
        )
    ):
        return EventOutcome.STALE_OR_DUPLICATE

    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_text_hash: str | None = None

    if prepared_text is not None and text_hash is not None:
        can_reuse = (
            existing is not None
            and existing.embedding is not None
            and existing.embedding_model == embedder.model_name
            and existing.embedding_text_hash == text_hash
        )
        if can_reuse:
            embedding = existing.embedding
        else:
            embedding = embedder.embed(prepared_text)
        embedding_model = embedder.model_name
        embedding_text_hash = text_hash

    with pool.connection() as conn:
        with conn.transaction():
            applied = upsert_company(
                conn,
                event,
                embedding=embedding,
                embedding_model=embedding_model,
                embedding_text_hash=embedding_text_hash,
            )

    return EventOutcome.APPLIED if applied else EventOutcome.STALE_OR_DUPLICATE


def _quarantine_processing_error(
    pool: ConnectionPool[Any],
    raw_body: bytes,
    message_id: str | None,
    exc: Exception,
) -> None:
    record = QuarantineRecord(
        message_id=message_id,
        registration_number=None,
        error_type="processing_error",
        error_message=f"{type(exc).__name__}: {exc}",
        raw_payload=raw_body.decode("utf-8", errors="replace"),
    )
    with pool.connection() as conn:
        with conn.transaction():
            record_ingestion_error(conn, record)


async def handle_message(
    message: AbstractIncomingMessage,
    pool: ConnectionPool[Any],
    embedder: Embedder,
    settings: Settings,
) -> EventOutcome | None:
    """Process one AMQP message with bounded backoff retries and commit-before-ACK.

    - Transient DB/network errors retry with exponential backoff before nack(requeue=True).
    - Deterministic non-DB processing errors are durably quarantined before ACK to prevent
      poison-message infinite requeue loops.
    """
    message_id = message.message_id
    max_attempts = max(1, settings.max_transient_retries + 1)

    for attempt in range(max_attempts):
        try:
            outcome = await asyncio.to_thread(
                process_event,
                message.body,
                message_id,
                pool,
                embedder,
            )
            await message.ack()
            return outcome
        except (psycopg.Error, PoolTimeout, OSError, TimeoutError) as exc:
            is_last_attempt = attempt >= max_attempts - 1
            delay = settings.retry_backoff_seconds * (2**attempt)
            logger.error(
                "Transient error processing message_id=%s (attempt %d/%d): %s: %s",
                message_id,
                attempt + 1,
                max_attempts,
                type(exc).__name__,
                exc,
            )
            if not is_last_attempt:
                await asyncio.sleep(delay)
                continue
            await message.nack(requeue=True)
            await asyncio.sleep(delay)
            return None
        except Exception as exc:
            logger.exception(
                "Deterministic processing error for message_id=%s; quarantining: %s",
                message_id,
                exc,
            )
            try:
                await asyncio.to_thread(
                    _quarantine_processing_error,
                    pool,
                    message.body,
                    message_id,
                    exc,
                )
            except (psycopg.Error, PoolTimeout, OSError, TimeoutError) as q_exc:
                logger.error(
                    "Failed to persist processing_error quarantine for message_id=%s: %s",
                    message_id,
                    q_exc,
                )
                await message.nack(requeue=True)
                await asyncio.sleep(settings.retry_backoff_seconds)
                return None
            await message.ack()
            return EventOutcome.QUARANTINED

    return None


async def run_consumer(settings: Settings | None = None) -> None:
    """Run the continuous RabbitMQ consumer until SIGINT/SIGTERM."""
    cfg = settings or Settings()

    logger.info("Applying idempotent database migrations")
    apply_migrations(cfg.effective_database_url)

    logger.info("Loading embedding model once: %s", cfg.embedding_model)
    embedder = await asyncio.to_thread(
        FastEmbedder,
        cfg.embedding_model,
        cfg.embedding_dim,
    )

    pool = create_pool(
        cfg.effective_database_url,
        min_size=cfg.db_pool_min_size,
        max_size=cfg.db_pool_max_size,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    stats = {outcome: 0 for outcome in EventOutcome}
    stats_failed = 0

    try:
        connection = await aio_pika.connect_robust(cfg.effective_rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=cfg.consumer_prefetch_count)
            # Queue owned by registry team: durable with no extra queue arguments.
            queue = await channel.declare_queue(cfg.rabbitmq_queue, durable=True)
            logger.info("Consuming from queue %s", cfg.rabbitmq_queue)

            async with queue.iterator() as queue_iter:
                consumer_task = asyncio.create_task(
                    _consume_loop(queue_iter, pool, embedder, cfg, stop_event, stats)
                )
                stop_task = asyncio.create_task(stop_event.wait())
                done, pending = await asyncio.wait(
                    {consumer_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done and not consumer_task.done():
                    logger.info("Shutdown requested; stopping consumer cleanly")
                    await queue_iter.close()
                    try:
                        await asyncio.wait_for(consumer_task, timeout=15.0)
                    except TimeoutError:
                        consumer_task.cancel()
                for task in pending:
                    if not task.done():
                        task.cancel()
                if consumer_task.done() and not consumer_task.cancelled():
                    stats_failed = consumer_task.result()
    finally:
        pool.close()
        logger.info(
            "Consumer stopped. applied=%d stale_or_duplicate=%d quarantined=%d transient_nacks=%d",
            stats[EventOutcome.APPLIED],
            stats[EventOutcome.STALE_OR_DUPLICATE],
            stats[EventOutcome.QUARANTINED],
            stats_failed,
        )


async def _consume_loop(
    queue_iter: Any,
    pool: ConnectionPool[Any],
    embedder: Embedder,
    cfg: Settings,
    stop_event: asyncio.Event,
    stats: dict[EventOutcome, int],
) -> int:
    transient_nacks = 0
    async for message in queue_iter:
        outcome = await handle_message(message, pool, embedder, cfg)
        if outcome is None:
            transient_nacks += 1
        else:
            stats[outcome] += 1
            total = sum(stats.values())
            if total % 500 == 0:
                logger.info(
                    "Progress: processed=%d applied=%d stale=%d quarantined=%d",
                    total,
                    stats[EventOutcome.APPLIED],
                    stats[EventOutcome.STALE_OR_DUPLICATE],
                    stats[EventOutcome.QUARANTINED],
                )
        if stop_event.is_set():
            break
    return transient_nacks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_consumer())


if __name__ == "__main__":
    main()
