"""Best-effort PII detection and masking.

Scope and honesty: regex detection is fast, dependency-free, and deterministic,
but it is NOT perfect — names, addresses, and free-form identifiers are
deliberately NOT detected (too many false positives to be useful). For
strong guarantees, layer this with a model-based detector ( presidio /
Azure AI Language PII) behind the same ``mask_pii`` interface.

Detected today: emails, phone numbers, SSNs, credit cards (Luhn-validated),
IPv4 addresses, and API keys / bearer tokens / JWTs — the highest-signal,
lowest-false-positive classes.

Two masking styles:
- ``replace``      — every match becomes a type placeholder like ``[EMAIL]``.
- ``pseudonymize`` — deterministic per-value aliases (``EMAIL_3f2a81c0``) that
  keep referential structure (same value → same alias) so prose like "email A
  belongs to phone B" survives, without revealing the value.
"""

import hashlib
import re
from collections.abc import Callable

# Order matters: secrets and cards are masked before the phone pattern, whose
# digit runs would otherwise overlap with them.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Candidate card numbers: 13–19 digits with optional spaces/dashes. Masked only
# when they pass Luhn validation, so order/reference numbers that merely look
# like card numbers stay intact.
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b")
_PHONE_RE = re.compile(
    # Entry guard excludes word chars and separators so we don't match digit
    # runs inside domains, order numbers, or pre-masked values.
    r"(?<![\w.-])(?:\+?1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}\b"
)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
# API keys, GitHub tokens, AWS access keys, JWTs (header.payload.signature).
_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|ey[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_SECRET_RE, "SECRET"),
    (_BEARER_RE, "SECRET"),
    (_EMAIL_RE, "EMAIL"),
    (_SSN_RE, "SSN"),
    (_IPV4_RE, "IP_ADDRESS"),
    (_PHONE_RE, "PHONE"),
)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _pseudonym(value: str, kind: str) -> str:
    tag = hashlib.sha256(value.strip().lower().encode()).hexdigest()[:8]
    return f"{kind}_{tag}"


def mask_pii(text: str, style: str = "replace") -> str:
    """Mask PII in ``text``.

    ``style`` is ``replace`` (placeholder per type) or ``pseudonymize``
    (deterministic per-value alias). Order matters: secrets and cards first so
    their digit runs are not re-matched by the phone pattern.
    """
    if not text:
        return ""

    def _sub(kind: str) -> Callable[[re.Match[str]], str]:
        def inner(m: re.Match[str]) -> str:
            return f"[{kind}]" if style == "replace" else _pseudonym(m.group(0), kind)

        return inner

    out = text

    def _mask_cards(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return (
                "[CREDIT_CARD]"
                if style == "replace"
                else _pseudonym(digits, "CREDIT_CARD")
            )
        return m.group(0)

    out = _CARD_CANDIDATE_RE.sub(_mask_cards, out)
    for pattern, kind in _PATTERNS:
        out = pattern.sub(_sub(kind), out)
    return out
