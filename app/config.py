"""Environment-based configuration for the consumer and API."""

from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings honoring existing POSTGRES_PORT and RABBITMQ_PORT conventions."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "companies"
    database_url: str | None = None

    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_url: str | None = None
    rabbitmq_queue: str = "company-events"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    consumer_prefetch_count: int = 16
    max_transient_retries: int = 3
    retry_backoff_seconds: float = 1.0

    db_pool_min_size: int = 1
    db_pool_max_size: int = 10

    @property
    def effective_database_url(self) -> str:
        """Return DATABASE_URL if provided, else build from POSTGRES_* variables."""
        if self.database_url:
            return self.database_url
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password, safe="")
        return (
            f"postgresql://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def effective_rabbitmq_url(self) -> str:
        """Return RABBITMQ_URL if provided, else build from RABBITMQ_* variables."""
        if self.rabbitmq_url:
            return self.rabbitmq_url
        user = quote(self.rabbitmq_user, safe="")
        password = quote(self.rabbitmq_password, safe="")
        return f"amqp://{user}:{password}@{self.rabbitmq_host}:{self.rabbitmq_port}/"
