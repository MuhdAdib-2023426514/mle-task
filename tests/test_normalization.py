"""Unit tests for event parsing, normalization, timestamp rules, and version ordering."""

import json
from datetime import UTC, datetime

import pytest

from app.normalization import (
    compute_embedding_text_hash,
    incoming_version_wins,
    normalize_registration_number,
    normalize_state,
    parse_raw_event,
    parse_updated_at,
    prepare_embedding_text,
)
from app.schemas import QuarantineRecord, StruckOffCompanyEvent, UpsertCompanyEvent


def _encode(obj: object) -> bytes:
    return json.dumps(obj).encode("utf-8")


def test_registration_number_normalizes_whitespace_and_case() -> None:
    assert normalize_registration_number(" syn-001550 ") == "SYN-001550"
    assert normalize_registration_number("SYN-001550") == "SYN-001550"
    with pytest.raises(ValueError):
        normalize_registration_number("   ")
    with pytest.raises(ValueError):
        normalize_registration_number(119)


def test_state_normalization_handles_casing_and_nulls() -> None:
    assert normalize_state("KUALA LUMPUR") == "Kuala Lumpur"
    assert normalize_state("  pulau   pinang ") == "Pulau Pinang"
    assert normalize_state("selangor") == "Selangor"
    assert normalize_state("") is None
    assert normalize_state(None) is None


def test_prepare_embedding_text_is_deterministic_and_handles_empty() -> None:
    assert prepare_embedding_text(None) is None
    assert prepare_embedding_text("   \t\n ") is None
    assert (
        prepare_embedding_text("  Operating   cafes\nand coffee shops  ")
        == "Operating cafes and coffee shops"
    )
    h1 = compute_embedding_text_hash("model-a", "Operating cafes")
    h2 = compute_embedding_text_hash("model-a", "Operating cafes")
    h3 = compute_embedding_text_hash("model-b", "Operating cafes")
    assert h1 == h2
    assert h1 != h3


def test_prepare_embedding_text_collapses_corrupted_repetitions_only() -> None:
    repeated = ("Hotel butik di Seremban " * 60).strip()
    assert prepare_embedding_text(repeated) == "Hotel butik di Seremban"

    # Also handles truncated final repeat
    truncated = "Ladang kelapa sawit " * 5 + "Ladang kelapa"
    assert prepare_embedding_text(truncated) == "Ladang kelapa sawit"

    # Normal sentences must remain untouched
    normal = "Operating cafes and coffee shops in the Klang Valley"
    assert prepare_embedding_text(normal) == normal
    two_repeats = "Coffee shop Coffee shop"
    assert prepare_embedding_text(two_repeats) == two_repeats


def test_parse_updated_at_variants_convert_to_utc() -> None:
    dt_myt, raw_myt = parse_updated_at("2026-08-14T09:15:02+08:00")
    assert dt_myt == datetime(2026, 8, 14, 1, 15, 2, tzinfo=UTC)
    assert raw_myt == "2026-08-14T09:15:02+08:00"

    dt_utc, _ = parse_updated_at("2026-08-14T01:15:02Z")
    assert dt_utc == dt_myt

    # Naive ISO timestamps from this Malaysian feed are interpreted as UTC+08:00
    dt_naive, _ = parse_updated_at("2026-08-14T09:16:02")
    assert dt_naive == datetime(2026, 8, 14, 1, 16, 2, tzinfo=UTC)
    assert dt_naive > dt_myt

    # Epoch milliseconds
    dt_ms, raw_ms = parse_updated_at(1786671038000)
    assert dt_ms == datetime(2026, 8, 14, 1, 30, 38, tzinfo=UTC)
    assert raw_ms == "1786671038000"

    # Missing / blank -> undated
    assert parse_updated_at(None) == (None, None)
    assert parse_updated_at("   ") == (None, None)

    with pytest.raises(ValueError):
        parse_updated_at(True)
    with pytest.raises(ValueError):
        parse_updated_at("not-a-timestamp")
    with pytest.raises(ValueError):
        parse_updated_at(12345)


def test_version_precedence_and_deterministic_tie_break() -> None:
    t1 = datetime(2026, 8, 14, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 8, 14, 2, 0, 0, tzinfo=UTC)

    # Newer beats older; older loses to newer
    assert incoming_version_wins(
        existing_updated_at=t1,
        existing_version_key="aaa",
        incoming_updated_at=t2,
        incoming_version_key="000",
    )
    assert not incoming_version_wins(
        existing_updated_at=t2,
        existing_version_key="000",
        incoming_updated_at=t1,
        incoming_version_key="fff",
    )

    # Dated beats undated; undated never overwrites dated
    assert incoming_version_wins(
        existing_updated_at=None,
        existing_version_key="fff",
        incoming_updated_at=t1,
        incoming_version_key="000",
    )
    assert not incoming_version_wins(
        existing_updated_at=t1,
        existing_version_key="000",
        incoming_updated_at=None,
        incoming_version_key="fff",
    )

    # Exact timestamp tie: higher key wins, identical key is a no-op
    assert incoming_version_wins(
        existing_updated_at=t1,
        existing_version_key="aaa",
        incoming_updated_at=t1,
        incoming_version_key="bbb",
    )
    assert not incoming_version_wins(
        existing_updated_at=t1,
        existing_version_key="bbb",
        incoming_updated_at=t1,
        incoming_version_key="aaa",
    )
    assert not incoming_version_wins(
        existing_updated_at=t1,
        existing_version_key="aaa",
        incoming_updated_at=t1,
        incoming_version_key="aaa",
    )


def test_parse_valid_upsert_ignores_extra_fields_and_normalizes() -> None:
    raw = _encode(
        {
            "type": "company-upserted",
            "registrationNumber": " syn-001466 ",
            "updatedAt": "2026-08-14T09:15:02+08:00",
            "source": "partner-api",
            "ingestedBy": "batch-1",
            "data": {
                "companyName": "  Indah Prima Makanan PLT ",
                "natureOfBusiness": "  ",
                "state": "SELANGOR",
                "industry": None,
            },
        }
    )
    parsed = parse_raw_event(raw, message_id="msg-1")
    assert isinstance(parsed, UpsertCompanyEvent)
    assert parsed.registration_number == "SYN-001466"
    assert parsed.company_name == "Indah Prima Makanan PLT"
    assert parsed.nature_of_business is None
    assert parsed.state == "Selangor"
    assert parsed.industry is None
    assert parsed.updated_at == datetime(2026, 8, 14, 1, 15, 2, tzinfo=UTC)


def test_parse_struck_off_event() -> None:
    raw = _encode(
        {
            "type": "company-struck-off",
            "registrationNumber": "SYN-001748",
            "updatedAt": "2026-08-14T16:00:33+08:00",
        }
    )
    parsed = parse_raw_event(raw, message_id="msg-so")
    assert isinstance(parsed, StruckOffCompanyEvent)
    assert parsed.registration_number == "SYN-001748"
    assert parsed.updated_at == datetime(2026, 8, 14, 8, 0, 33, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw_body", "expected_error"),
    [
        (b'{"type": "company-upserted", "data": {"companyName": "Bes', "invalid_json"),
        (b"null", "invalid_envelope"),
        (
            _encode({"type": "company-upserted", "data": {"companyName": "No Reg"}}),
            "missing_registration_number",
        ),
        (
            _encode(
                {
                    "type": "company-upserted",
                    "registrationNumber": 119,
                    "data": {"companyName": "X"},
                }
            ),
            "invalid_registration_number",
        ),
        (_encode({"type": "company-upserted", "registrationNumber": "SYN-000001"}), "invalid_data"),
        (
            _encode(
                {"type": "company-upserted", "registrationNumber": "SYN-000001", "data": "k=v"}
            ),
            "invalid_data",
        ),
        (
            _encode({"type": "unknown-event", "registrationNumber": "SYN-000001"}),
            "unsupported_event_type",
        ),
        (
            _encode(
                {
                    "type": "company-upserted",
                    "registrationNumber": "SYN-000001",
                    "updatedAt": "bad-date",
                    "data": {"companyName": "X"},
                }
            ),
            "invalid_timestamp",
        ),
    ],
)
def test_malformed_events_are_quarantined(raw_body: bytes, expected_error: str) -> None:
    parsed = parse_raw_event(raw_body, message_id="bad-msg")
    assert isinstance(parsed, QuarantineRecord)
    assert parsed.error_type == expected_error
    assert parsed.message_id == "bad-msg"
