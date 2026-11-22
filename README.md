# OneCredit MLE Take-Home: Company Recommendation Service

Content-based **"Similar companies"** recommendation service built for OneCredit's portal (`mle-v2026.11`). It consumes noisy, multilingual, at-least-once company events from RabbitMQ (`company-events`), embeds business descriptions locally on CPU with an open-source ONNX model, stores company state and 384-dimensional vectors in PostgreSQL (`pgvector`), and serves low-latency nearest-neighbor recommendations via FastAPI.

See [`DESIGN.md`](DESIGN.md) for architectural decisions, messy-feed policies, measured multilingual vs. English-only retrieval results, 10x scaling plan, limitations, and AI-usage disclosure.

---

## Architecture Overview

```text
producer/publish.py (data/feed.jsonl)
        │  durable queue: company-events (no extra args)
        ▼
    RabbitMQ  ──►  app.consumer (aio-pika + FastEmbed ONNX CPU, loaded once)
                        │  atomic conditional upsert + commit-before-ACK
                        │  malformed events -> ingestion_errors
                        ▼
               PostgreSQL + pgvector
                        ▲
                        │  exact cosine search (<=>), inference-free
                        │
                app.api (FastAPI)  ◄──  OneCredit Portal / curl
```

---

## Prerequisites

- **Docker** (with Docker Compose) for PostgreSQL (`pgvector/pgvector:0.8.7-pg16`) and RabbitMQ (`rabbitmq:3.13.7-management`)
- **[`uv`](https://docs.astral.sh/uv/)** (manages Python 3.14 and locked dependencies automatically)
- **[`just`](https://github.com/casey/just)** *(optional — every `just` target has a direct command equivalent below)*

> **First-run model download:** On its first start, the consumer downloads the open-source quantized ONNX model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~220 MB) via `fastembed` into the local Hugging Face cache. No paid API keys or external services are required.

---

## Quick Start

### 1. Install dependencies & start infrastructure

```bash
just setup          # or: uv sync --locked
just up             # or: docker compose up -d --wait
```

### 2. Apply database schema migration (one-time, idempotent)

```bash
uv run python -m app.database
```

*(The consumer also runs this idempotent migration on startup.)*

### 3. Start the consumer (Terminal A)

```bash
uv run python -m app.consumer
```

### 4. Start the similarity API (Terminal B)

```bash
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000
```

### 5. Publish the registry feed (Terminal C)

```bash
just publish        # or: uv run python -m producer.publish
```

Expected producer output: `published 2518 messages to company-events`.

---

## Configuration (Environment Variables)

All settings have defaults matching `docker-compose.yml` and honor `POSTGRES_PORT`, `RABBITMQ_PORT`, and `RABBITMQ_UI_PORT`:

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | Ignored if `DATABASE_URL` is set |
| `POSTGRES_PORT` | `5432` | Matches `docker-compose.yml` |
| `POSTGRES_USER` | `postgres` | |
| `POSTGRES_PASSWORD` | `postgres` | |
| `POSTGRES_DB` | `companies` | |
| `DATABASE_URL` | *(unset)* | Explicit override (e.g. `postgresql://postgres:postgres@localhost:5432/companies`) |
| `RABBITMQ_HOST` | `localhost` | Ignored if `RABBITMQ_URL` is set |
| `RABBITMQ_PORT` | `5672` | Matches `docker-compose.yml` and `producer/publish.py` |
| `RABBITMQ_USER` | `guest` | |
| `RABBITMQ_PASSWORD` | `guest` | |
| `RABBITMQ_URL` | *(unset)* | Explicit override (e.g. `amqp://guest:guest@localhost:5672/`) |
| `RABBITMQ_QUEUE` | `company-events` | Owned by producer (`durable=True`, no extra arguments) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384-dim CPU ONNX multilingual model via `fastembed` |
| `EMBEDDING_DIM` | `384` | Must match the `vector(384)` column |
| `CONSUMER_PREFETCH_COUNT` | `16` | RabbitMQ QoS prefetch limit |
| `MAX_TRANSIENT_RETRIES` | `3` | Bounded retries with exponential backoff before `nack(requeue=True)` |
| `RETRY_BACKOFF_SECONDS` | `1.0` | Base retry backoff in seconds |

RabbitMQ Management UI is available at `http://localhost:15672` (default credentials `guest` / `guest`, or `RABBITMQ_UI_PORT`).

---

## API Usage

### Endpoints

- `GET /health` — verifies PostgreSQL connectivity (`200` or `503`).
- `GET /companies/{registration_number}/similar?limit=5`
  - Normalizes `registration_number` via `strip().upper()`.
  - `limit` is validated and bounded to `1..20` (default `5`; invalid values return `422`).
  - Excludes the query company, `struck_off` companies, and companies with `embedding IS NULL` from semantic candidates; ties in cosine similarity break by same `state`, same `industry`, and `registration_number ASC`.
  - Each recommendation includes `"matchType": "semantic"` (with cosine `similarity`) or, when a known company lacks a business description but has a real `industry`, `"matchType": "industry_fallback"` with `"similarity": null` (never fabricating cosine scores).
  - Returns `404` for unknown companies, `200` with `recommendations: []` and an explicit `message` when both description and industry are missing (or no active candidates exist), and `503` if PostgreSQL is unavailable.

### Example `curl` requests & observed output

```bash
# English coffee shop query
curl -s 'http://127.0.0.1:8000/companies/SYN-001550/similar?limit=5' | python3 -m json.tool
```

Observed response after ingesting `data/feed.jsonl`:

```json
{
  "registrationNumber": "SYN-001550",
  "companyName": "Sinar Teguh Bakery Sdn. Bhd.",
  "industry": "Food & Beverage",
  "state": "Selangor",
  "status": "active",
  "recommendations": [
    {
      "registrationNumber": "SYN-000539",
      "companyName": "Seri Sejahtera Food Sdn. Bhd.",
      "industry": "Food & Beverage",
      "state": "Johor",
      "similarity": 1.0,
      "matchType": "semantic"
    },
    {
      "registrationNumber": "SYN-000263",
      "companyName": "Gemilang Delima Food Sdn. Bhd.",
      "industry": "Food & Beverage",
      "state": "Johor",
      "similarity": 0.9497,
      "matchType": "semantic"
    },
    {
      "registrationNumber": "SYN-001650",
      "companyName": "Everbright Pacific Food Sdn. Bhd.",
      "industry": "Food & Beverage",
      "state": "Kuala Lumpur",
      "similarity": 0.9497,
      "matchType": "semantic"
    },
    {
      "registrationNumber": "SYN-000344",
      "companyName": "Harmoni Maju Kopi Sdn. Bhd.",
      "industry": "Food & Beverage",
      "state": "Kedah",
      "similarity": 0.8618,
      "matchType": "semantic"
    },
    {
      "registrationNumber": "SYN-000463",
      "companyName": "Sinar Unity Kopi PLT",
      "industry": "Food & Beverage",
      "state": "Kuala Lumpur",
      "similarity": 0.8618,
      "matchType": "semantic"
    }
  ],
  "message": null
}
```

Additional checks:

```bash
# Health check
curl -s http://127.0.0.1:8000/health

# Malay description query ("Hotel butik di Seremban")
curl -s 'http://127.0.0.1:8000/companies/SYN-000506/similar?limit=5'

# Missing description with real industry (returns 200 with labeled industry_fallback & similarity: null)
curl -s 'http://127.0.0.1:8000/companies/SYN-000014/similar?limit=3'

# Missing both description and industry (returns 200 with recommendations: [] + explicit message)
curl -s 'http://127.0.0.1:8000/companies/SYN-000048/similar?limit=3'

# Unknown company (returns 404)
curl -i 'http://127.0.0.1:8000/companies/SYN-999999/similar'
```

---

## Verifying Ingestion, Queue Drain & Replay Safety

Check that RabbitMQ has drained completely (`0` ready, `0` unacknowledged):

```bash
docker compose exec -T rabbitmq rabbitmqctl list_queues -q name messages_ready messages_unacknowledged
```

Inspect registry coverage, embedding coverage, struck-off status, and quarantined errors in PostgreSQL:

```bash
docker compose exec -T postgres psql -U postgres -d companies -c "
SELECT
  COUNT(*) AS total_companies,
  COUNT(embedding) AS with_embedding,
  COUNT(*) FILTER (WHERE embedding IS NULL) AS without_embedding,
  COUNT(*) FILTER (WHERE status = 'active') AS active_companies,
  COUNT(*) FILTER (WHERE status = 'struck_off') AS struck_off_companies
FROM companies;
SELECT error_type, COUNT(*) FROM ingestion_errors GROUP BY error_type ORDER BY COUNT(*) DESC, error_type;
"
```

Observed baseline after ingesting `data/feed.jsonl`:
- `total_companies = 2000` (`1992` active, `8` struck off)
- `with_embedding = 1961`, `without_embedding = 39` (`NULL` vectors preserved, zero lost companies)
- `ingestion_errors = 25` distinct quarantined payloads (covering all 30 invalid messages; 6 identical `null` payloads share one SHA-256 `message_id` and deduplicate cleanly)

To verify **replay safety**, run `just publish` (or `uv run python -m producer.publish`) a second time while the consumer is running: all valid messages short-circuit as `stale_or_duplicate` (`applied=0`), malformed messages hit `ON CONFLICT DO NOTHING`, and all 2,000 company rows and 25 error rows remain identical.

---

## Tests, Linting & Evaluation

Integration tests automatically create and use an isolated `companies_test` database and temporary RabbitMQ queues so they never modify the primary `companies` database or `company-events` queue.

```bash
# Fast unit tests only (no Docker or model download required)
uv run pytest -m "not integration" -q

# Full test suite (unit + isolated Postgres/pgvector + FastAPI + RabbitMQ + ONNX smoke test)
uv run pytest -q

# Linting and formatting checks
uv run ruff check .
uv run ruff format --check .

# Optional P2 comparison: multilingual vs. English-only model on active DB catalog
uv run python -m scripts.eval_models

# Optional scale benchmark: exact cosine vs. pgvector HNSW (2k / 20k / 50k) & ONNX batching
uv run python -m scripts.benchmark_scale
```

---

## Shutdown & Clean Reset

- Stop the consumer and API cleanly with `Ctrl+C` (`SIGINT`) or `SIGTERM`.
- Stop containers while keeping volumes: `docker compose down`
- Stop containers and **wipe all Postgres/RabbitMQ data** for a clean slate: `just down` (or `docker compose down -v`)

---

## Project Layout

```text
app/
  config.py          # env-based configuration (POSTGRES_PORT / RABBITMQ_PORT / URL overrides)
  schemas.py         # internal event dataclasses and FastAPI Pydantic response schemas
  normalization.py   # pure parsing, ID/state/timestamp normalization, and version rules
  embeddings.py      # FastEmbed ONNX CPU wrapper loaded once per consumer worker
  database.py        # migration runner, connection pool, conditional upserts, pgvector queries
  consumer.py        # RabbitMQ consumer with commit-before-ACK, quarantine, and bounded backoff
  api.py             # inference-free FastAPI retrieval service
migrations/
  001_initial.sql    # pgvector extension, companies table, ingestion_errors table
scripts/
  eval_models.py     # reproducible multilingual vs. English-only retrieval evaluation
  benchmark_scale.py # reproducible 1x/10x/25x exact-vs-HNSW & ONNX batching benchmark
tests/
  conftest.py        # FakeEmbedder and isolated companies_test Postgres fixtures
  test_normalization.py
  test_versioning.py
  test_api.py
  test_integration.py
DESIGN.md            # architecture, messy-feed policies, trade-offs, P2 eval, 10x scaling, AI disclosure
```
