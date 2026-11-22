CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS companies (
    registration_number TEXT PRIMARY KEY,
    company_name TEXT,
    industry TEXT,
    state TEXT,
    nature_of_business TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'struck_off')),
    updated_at TIMESTAMPTZ,
    raw_updated_at TEXT,
    event_version_key TEXT NOT NULL,
    last_message_id TEXT,
    embedding vector(384),
    embedding_model TEXT,
    embedding_text_hash TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_active_with_embedding
    ON companies (status)
    WHERE status = 'active' AND embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS ingestion_errors (
    id BIGSERIAL PRIMARY KEY,
    message_id TEXT,
    registration_number TEXT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    raw_payload TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (message_id, error_type)
);
