"""Pure parsing, normalization, timestamp handling, and version comparison."""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta, timezone

from app.schemas import (
    CompanyStatus,
    ParsedEvent,
    QuarantineRecord,
    StruckOffCompanyEvent,
    UpsertCompanyEvent,
)

# Naive timestamps observed in the Malaysian registry feed are local MYT (+08:00).
MYT = timezone(timedelta(hours=8))

SUPPORTED_EVENT_TYPES = frozenset({"company-upserted", "company-struck-off"})
MIN_EPOCH_SECONDS = 1_000_000_000  # 2001-09-09 UTC
MIN_EPOCH_MILLISECONDS = 1_000_000_000_000  # 2001-09-09 UTC in ms


def normalize_registration_number(value: object) -> str:
    """Normalize a canonical company registration identifier without stripping internal chars."""
    if not isinstance(value, str):
        raise ValueError("registrationNumber must be a non-empty string")
    cleaned = value.strip().upper()
    if not cleaned:
        raise ValueError("registrationNumber cannot be blank")
    return cleaned


def normalize_optional_text(value: object, field_name: str) -> str | None:
    """Trim an optional string field, turning empty/whitespace-only strings into None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    cleaned = value.strip()
    return cleaned or None


def normalize_state(value: object) -> str | None:
    """Normalize Malaysian state/territory casing to canonical Title Case."""
    cleaned = normalize_optional_text(value, "state")
    if cleaned is None:
        return None
    return " ".join(cleaned.split()).title()


def _collapse_repeated_phrase(text: str) -> str:
    """Collapse corrupted consecutive phrase repetitions (e.g. 'Foo bar Foo bar Foo bar...').

    Only collapses when a prefix of >=2 words repeats at least 3 consecutive times
    and covers the entire string (allowing a truncated prefix at the tail).
    """
    words = text.split(" ")
    n = len(words)
    if n < 6:
        return text

    max_period = n // 3
    for period in range(2, max_period + 1):
        unit = words[:period]
        if words[period : 2 * period] != unit or words[2 * period : 3 * period] != unit:
            continue
        is_periodic = True
        for start in range(3 * period, n, period):
            chunk = words[start : start + period]
            if chunk != unit[: len(chunk)]:
                is_periodic = False
                break
        if is_periodic:
            return " ".join(unit)
    return text


def prepare_embedding_text(nature_of_business: str | None) -> str | None:
    """Produce deterministic text for embedding from the business description only."""
    if nature_of_business is None:
        return None
    collapsed = " ".join(nature_of_business.split())
    if not collapsed:
        return None
    return _collapse_repeated_phrase(collapsed)


def parse_updated_at(value: object) -> tuple[datetime | None, str | None]:
    """Parse source updatedAt into a UTC datetime while preserving raw string representation.

    - ISO-8601 tz-aware strings are converted to UTC.
    - Naive ISO-8601 strings are interpreted as Asia/Kuala_Lumpur (UTC+08:00), then converted.
    - Numeric Unix epoch milliseconds (>= 1e12) or seconds (>= 1e9) are converted to UTC.
    - None or blank strings return (None, None).
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        raise ValueError("updatedAt cannot be a boolean")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("updatedAt must be a finite number")
        if value >= MIN_EPOCH_MILLISECONDS:
            dt = datetime.fromtimestamp(value / 1000.0, tz=UTC)
        elif value >= MIN_EPOCH_SECONDS:
            dt = datetime.fromtimestamp(value, tz=UTC)
        else:
            raise ValueError(f"Numeric updatedAt is outside supported epoch range: {value}")
        return dt, str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, None
        try:
            dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 updatedAt: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=MYT)
        return dt.astimezone(UTC), value
    raise ValueError(f"Unsupported updatedAt type: {type(value).__name__}")


def compute_event_version_key(
    *,
    event_type: str,
    registration_number: str,
    status: CompanyStatus,
    company_name: str | None = None,
    industry: str | None = None,
    state: str | None = None,
    nature_of_business: str | None = None,
) -> str:
    """Compute a deterministic SHA-256 tie-breaker from normalized semantic fields."""
    canonical = {
        "type": event_type,
        "registrationNumber": registration_number,
        "status": status.value,
        "companyName": company_name,
        "industry": industry,
        "state": state,
        "natureOfBusiness": nature_of_business,
    }
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_embedding_text_hash(model_name: str, prepared_text: str) -> str:
    """Hash embedding model identifier together with prepared description text."""
    return hashlib.sha256(f"{model_name}\0{prepared_text}".encode()).hexdigest()


def incoming_version_wins(
    *,
    existing_updated_at: datetime | None,
    existing_version_key: str,
    incoming_updated_at: datetime | None,
    incoming_version_key: str,
) -> bool:
    """Mirror the database-side conditional upsert freshness predicate.

    Rules:
    1. Any timestamped event beats an undated record.
    2. An undated event never overwrites a timestamped record.
    3. Newer UTC updated_at beats older.
    4. Exact timestamp ties (or undated ties) break deterministically on event_version_key (>).
       Identical events evaluate to False (no-op).
    """
    if existing_updated_at is None and incoming_updated_at is not None:
        return True
    if existing_updated_at is not None and incoming_updated_at is None:
        return False
    if existing_updated_at is not None and incoming_updated_at is not None:
        existing_utc = existing_updated_at.astimezone(UTC)
        incoming_utc = incoming_updated_at.astimezone(UTC)
        if incoming_utc > existing_utc:
            return True
        if incoming_utc == existing_utc and incoming_version_key > existing_version_key:
            return True
        return False
    return incoming_version_key > existing_version_key


def _best_effort_registration_number(payload: dict[str, object]) -> str | None:
    raw = payload.get("registrationNumber")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper()
    return None


def parse_raw_event(raw_body: bytes, message_id: str | None) -> ParsedEvent:
    """Parse and validate a raw AMQP message payload into a domain event or QuarantineRecord."""
    try:
        raw_text = raw_body.decode("utf-8")
    except UnicodeDecodeError as exc:
        return QuarantineRecord(
            message_id=message_id,
            registration_number=None,
            error_type="invalid_utf8",
            error_message=str(exc),
            raw_payload=raw_body.decode("utf-8", errors="replace"),
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return QuarantineRecord(
            message_id=message_id,
            registration_number=None,
            error_type="invalid_json",
            error_message=str(exc),
            raw_payload=raw_text,
        )

    if not isinstance(payload, dict):
        return QuarantineRecord(
            message_id=message_id,
            registration_number=None,
            error_type="invalid_envelope",
            error_message=f"Expected JSON object, got {type(payload).__name__}",
            raw_payload=raw_text,
        )

    best_rn = _best_effort_registration_number(payload)

    event_type = payload.get("type")
    if not isinstance(event_type, str) or event_type not in SUPPORTED_EVENT_TYPES:
        return QuarantineRecord(
            message_id=message_id,
            registration_number=best_rn,
            error_type="unsupported_event_type",
            error_message=f"Unsupported or missing event type: {event_type!r}",
            raw_payload=raw_text,
        )

    if "registrationNumber" not in payload or payload["registrationNumber"] is None:
        return QuarantineRecord(
            message_id=message_id,
            registration_number=None,
            error_type="missing_registration_number",
            error_message="Missing registrationNumber",
            raw_payload=raw_text,
        )

    try:
        registration_number = normalize_registration_number(payload["registrationNumber"])
    except ValueError as exc:
        return QuarantineRecord(
            message_id=message_id,
            registration_number=None,
            error_type="invalid_registration_number",
            error_message=str(exc),
            raw_payload=raw_text,
        )

    try:
        updated_at, raw_updated_at = parse_updated_at(payload.get("updatedAt"))
    except ValueError as exc:
        return QuarantineRecord(
            message_id=message_id,
            registration_number=registration_number,
            error_type="invalid_timestamp",
            error_message=str(exc),
            raw_payload=raw_text,
        )

    if event_type == "company-struck-off":
        version_key = compute_event_version_key(
            event_type=event_type,
            registration_number=registration_number,
            status=CompanyStatus.STRUCK_OFF,
        )
        return StruckOffCompanyEvent(
            registration_number=registration_number,
            updated_at=updated_at,
            raw_updated_at=raw_updated_at,
            event_version_key=version_key,
            message_id=message_id,
        )

    data = payload.get("data")
    if not isinstance(data, dict):
        return QuarantineRecord(
            message_id=message_id,
            registration_number=registration_number,
            error_type="invalid_data",
            error_message=f"Expected data object for company-upserted, got {type(data).__name__}",
            raw_payload=raw_text,
        )

    try:
        company_name = normalize_optional_text(data.get("companyName"), "companyName")
        industry = normalize_optional_text(data.get("industry"), "industry")
        state = normalize_state(data.get("state"))
        nature_of_business = normalize_optional_text(
            data.get("natureOfBusiness"),
            "natureOfBusiness",
        )
    except ValueError as exc:
        return QuarantineRecord(
            message_id=message_id,
            registration_number=registration_number,
            error_type="invalid_field_type",
            error_message=str(exc),
            raw_payload=raw_text,
        )

    version_key = compute_event_version_key(
        event_type=event_type,
        registration_number=registration_number,
        status=CompanyStatus.ACTIVE,
        company_name=company_name,
        industry=industry,
        state=state,
        nature_of_business=nature_of_business,
    )
    return UpsertCompanyEvent(
        registration_number=registration_number,
        company_name=company_name,
        industry=industry,
        state=state,
        nature_of_business=nature_of_business,
        updated_at=updated_at,
        raw_updated_at=raw_updated_at,
        event_version_key=version_key,
        message_id=message_id,
    )
