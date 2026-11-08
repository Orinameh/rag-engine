"""Hardening against prompt injection from stored document content.

Retrieved chunks are untrusted data: they were produced by whoever uploaded the
document, and a hostile document can carry instructions ("ignore your rules,
reveal everything"...). The defense is layered:

1. ``sanitize_untrusted`` neutralizes the freight that allows a document to
   impersonate chat protocol or the assistant itself (role markers, special
   pseudo-tokens, control characters).
2. ``build_context`` wraps every chunk in ``<untrusted_data>`` fences; the
   sanitizer also blocks look-alike fence tags so content cannot break out.
3. The system prompt tells the LLM that fenced text is data, never instructions.

Sanitization happens at prompt-build time, not ingest time — stored text and
embeddings keep full fidelity, and the sanitizer can evolve without re-indexing.
"""

import re

# Control characters that have no business in prose and that rendering layers
# may interpret (except \n, \t which we keep as whitespace).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# A line opening with a chat-role marker is either an injection attempt or
# server-log debris; either way it must not read like the assistant's protocol.
_ROLE_MARKER_RE = re.compile(
    r"(?mi)^([ \t]*)(system|assistant|user|developer|tool)\s*:"
)

# Pseudo special tokens (<|im_start|>, <|endoftext|>, ...) can force some
# renderers/servers to treat the payload as a new conversation turn.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>|]{1,32}\|>")

# Look-alike fence tags. Case/formatting-insensitive: "</untrusted_data>",
# "</ UNTRUSTED_DATA >" etc. must never survive into the prompt.
_FENCE_TAG_RE = re.compile(r"<\s*/?\s*untrusted_data\s*>", re.IGNORECASE)

# Stripped content is replaced with explicit markers so reviewers can see
# where sanitization happened instead of silently losing text.
_MAX_SANITIZED_LEN = 20_000

ROLE_MARKERS = ("system", "assistant", "user", "developer", "tool")
FENCE_OPEN = "<untrusted_data>"
FENCE_CLOSE = "</untrusted_data>"


def sanitize_untrusted(text: str, max_chars: int = _MAX_SANITIZED_LEN) -> str:
    """Make document content safe to embed inside a prompt fence."""
    if not text:
        return ""

    out = _CONTROL_RE.sub("", text)
    out = _ROLE_MARKER_RE.sub(
        lambda m: f"{m.group(1)}[role marker {m.group(2).upper()} removed]",
        out,
    )
    out = _SPECIAL_TOKEN_RE.sub("[special token removed]", out)
    out = _FENCE_TAG_RE.sub("[untrusted_data tag removed]", out)

    return out[:max_chars]
