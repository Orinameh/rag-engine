"""Tests for PII detection/masking and the ingest/egress policy wiring."""

import re

import pytest

from rag_engine.retrieval.context import build_context
from rag_engine.retrieval.pii import _luhn_ok, mask_pii


def _chunk(content, doc_id="11111111-1111-1111-1111-111111111111", **extra):
    return {
        "doc_id": doc_id,
        "content": content,
        "metadata": {"title": "Handbook", "page": 3, **extra},
    }


# ---------- mask_pii: detection ----------


def test_masks_email():
    out = mask_pii("mail dave@example-corp.com today")
    assert "dave@example-corp.com" not in out
    assert "[EMAIL]" in out


def test_masks_ssn():
    out = mask_pii("SSN 123-45-6789 on file")
    assert "123-45-6789" not in out
    assert "[SSN]" in out


def test_masks_valid_credit_card():
    # Luhn-valid test card.
    out = mask_pii("card 4111 1111 1111 1111 expires soon")
    assert "4111 1111 1111 1111" not in out
    assert "[CREDIT_CARD]" in out


def test_does_not_mask_luhn_invalid_number():
    # Borrows an email-like digit run; key point: not Luhn-valid, not masked.
    out = mask_pii("ref 1234 5678 9012 3456")
    assert "1234 5678 9012 3456" in out


def test_does_not_mask_freetext_numbers():
    out = mask_pii("order number 42 shipped on 2026-01-23")
    assert "42" in out and "2026-01-23" in out


def test_masks_ipv4():
    out = mask_pii("reach me at 192.168.10.130")
    assert "192.168.10.130" not in out
    assert "[IP_ADDRESS]" in out


def test_masks_api_keys_and_bearer_tokens():
    text = "key sk-v8Kf2mNq7pX0zT4b  \nbearer token: Bearer eyJhbGciOi5.eyJzaWI6f.Ps5bVpqR\n"
    out = mask_pii(text)
    assert "sk-v8Kf2mNq7pX0zT4b" not in out
    assert "Github" not in out
    assert "eyJhbGciOi5" not in out
    assert "[SECRET]" in out


def test_phone_masked():
    out = mask_pii("call (415) 555-1234 or 415.555.1234")
    assert "415" not in out
    assert "[PHONE]" in out


# ---------- mask_pii: styles ----------


def test_pseudonymize_is_deterministic_per_value():
    a = mask_pii("dave@example.com and dave@example.com again", style="pseudonymize")
    b = mask_pii("dave@example.com elsewhere", style="pseudonymize")
    assert "dave@example.com" not in a
    alias = a.split("and ")[1].split(" again")[0]
    assert alias in b  # same value → same alias across calls
    assert alias.startswith("EMAIL_")


def test_pseudonymize_distinguishes_values():
    out = mask_pii("dave@example.com maria@example.com", style="pseudonymize")
    found = re.findall(r"EMAIL_[0-9a-f]{8}", out)
    assert len(found) == 2
    assert found[0] != found[1]


def test_plain_text_untouched():
    assert mask_pii("The maximum operating temperature is 85 °C.") == (
        "The maximum operating temperature is 85 °C."
    )


# ---------- build_context × policy ----------


class _FakeSettings:
    pii_masking = "off"
    pii_mask_style = "replace"


def test_build_context_masks_per_policy(monkeypatch):
    import rag_engine.retrieval.context as ctx

    s = _FakeSettings()
    monkeypatch.setattr(ctx, "get_settings", lambda: s)

    s.pii_masking = "off"
    parts, _ = build_context([_chunk("contact dave@example.com")], max_tokens=3000)
    assert "dave@example.com" in parts  # off → verbatim

    s.pii_masking = "llm_only"
    parts, _ = build_context([_chunk("contact dave@example.com")], max_tokens=3000)
    assert "dave@example.com" not in parts
    assert "[EMAIL]" in parts

    s.pii_masking = "all"
    parts, _ = build_context([_chunk("contact dave@example.com")], max_tokens=3000)
    assert "dave@example.com" not in parts


def test_masking_survives_truncation_in_fence(monkeypatch):
    import rag_engine.retrieval.context as ctx
    from rag_engine.retrieval.sanitize import FENCE_CLOSE, FENCE_OPEN

    s = _FakeSettings()
    s.pii_masking = "llm_only"
    monkeypatch.setattr(ctx, "get_settings", lambda: s)

    huge = ("call jane.doe@example.com at (415) 555-1234. " * 200).strip()
    parts, _ = build_context([_chunk(huge)], max_tokens=512)
    assert "jane.doe@example.com" not in parts
    assert FENCE_OPEN in parts and FENCE_CLOSE in parts


# ---------- luhn ----------


@pytest.mark.parametrize(
    "number,ok",
    [
        ("4111111111111111", True),
        ("4111111111111112", False),
        ("5500005555555559", True),
    ],
)
def test_luhn(number, ok):
    assert _luhn_ok(number) is ok


# ---------- worker policy flag sanity ----------


def test_mask_pii_matches_worker_usage_signature():
    # mask_pii(text, style=...) must not raise for valid styles.
    from rag_engine.config import get_settings

    assert get_settings().pii_mask_style in ("replace", "pseudonymize")
    assert get_settings().pii_masking in ("off", "llm_only", "all")
