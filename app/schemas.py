"""Internal domain models and API response schemas."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EventOutcome(StrEnum):
    """Result of processing a single RabbitMQ delivery."""

    APPLIED = "applied"
    STALE_OR_DUPLICATE = "stale_or_duplicate"
    QUARANTINED = "quarantined"


class CompanyStatus(StrEnum):
    """Lifecycle status of a company in the registry."""

    ACTIVE = "active"
    STRUCK_OFF = "struck_off"


class MatchType(StrEnum):
    """Indicates how a recommendation candidate was matched."""

    SEMANTIC = "semantic"
    INDUSTRY_FALLBACK = "industry_fallback"


@dataclass(frozen=True, slots=True)
class UpsertCompanyEvent:
    """Normalized company-upserted event ready for embedding and persistence."""

    registration_number: str
    company_name: str | None
    industry: str | None
    state: str | None
    nature_of_business: str | None
    updated_at: datetime | None
    raw_updated_at: str | None
    event_version_key: str
    message_id: str | None


@dataclass(frozen=True, slots=True)
class StruckOffCompanyEvent:
    """Normalized company-struck-off event."""

    registration_number: str
    updated_at: datetime | None
    raw_updated_at: str | None
    event_version_key: str
    message_id: str | None


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """Structurally invalid or unprocessable event to persist in ingestion_errors."""

    message_id: str | None
    registration_number: str | None
    error_type: str
    error_message: str
    raw_payload: str


@dataclass(frozen=True, slots=True)
class StoredCompanyVersion:
    """Version and embedding metadata read from Postgres for optimization checks."""

    registration_number: str
    status: CompanyStatus
    updated_at: datetime | None
    event_version_key: str
    has_metadata: bool
    embedding: list[float] | None
    embedding_model: str | None
    embedding_text_hash: str | None


ParsedEvent = UpsertCompanyEvent | StruckOffCompanyEvent | QuarantineRecord


@dataclass(frozen=True, slots=True)
class CompanyProfile:
    """Stored company metadata used by the retrieval API."""

    registration_number: str
    company_name: str | None
    industry: str | None
    state: str | None
    nature_of_business: str | None
    status: CompanyStatus
    has_embedding: bool


@dataclass(frozen=True, slots=True)
class SimilarCompanyRow:
    """Single candidate returned by pgvector cosine search or metadata fallback."""

    registration_number: str
    company_name: str | None
    industry: str | None
    state: str | None
    similarity: float | None
    match_type: MatchType = MatchType.SEMANTIC


class SimilarCompanyItem(BaseModel):
    """Portal card representation for a single recommended company."""

    model_config = ConfigDict(populate_by_name=True)

    registration_number: str = Field(alias="registrationNumber")
    company_name: str | None = Field(alias="companyName")
    industry: str | None = None
    state: str | None = None
    similarity: float | None = None
    match_type: MatchType = Field(default=MatchType.SEMANTIC, alias="matchType")


class SimilarCompaniesResponse(BaseModel):
    """Response payload for GET /companies/{registration_number}/similar."""

    model_config = ConfigDict(populate_by_name=True)

    registration_number: str = Field(alias="registrationNumber")
    company_name: str | None = Field(alias="companyName")
    industry: str | None = None
    state: str | None = None
    status: CompanyStatus
    recommendations: list[SimilarCompanyItem]
    message: str | None = None


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str
    database: str
