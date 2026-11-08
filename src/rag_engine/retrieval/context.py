import tiktoken

from rag_engine.config import get_settings
from rag_engine.retrieval.pii import mask_pii
from rag_engine.retrieval.sanitize import FENCE_CLOSE, FENCE_OPEN, sanitize_untrusted

_ENC = tiktoken.get_encoding("cl100k_base")


def _fence(content: str) -> str:
    """Wrap untrusted document content in a data fence.

    Inside the fence the content has been sanitized, so it cannot contain live
    fence tags (see sanitize_untrusted) and cannot impersonate the protocol
    that separates data from instructions in the conversation prompt.
    """
    return f"{FENCE_OPEN}\n{content}\n{FENCE_CLOSE}"


def _prepare(content: str) -> str:
    """Sanitize against injection and mask PII per the configured policy."""
    out = sanitize_untrusted(content)
    mode = get_settings().pii_masking
    if mode != "off":
        return mask_pii(out, style=get_settings().pii_mask_style)
    return out


def build_context(chunks: list[dict], max_tokens: int = 3000) -> tuple[str, list[dict]]:
    parts: list[str] = []
    citations: list[dict] = []
    used = 0
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        header = (
            f"[SOURCE {i}] doc_id={c['doc_id']} "
            f"title={meta.get('title', 'untitled')} page={meta.get('page')}"
        )
        # Hostile content is neutralized and PII masked before it reaches the prompt.
        block = f"{header}\n{_fence(_prepare(c['content']))}"
        tokens = len(_ENC.encode(block))
        if used + tokens > max_tokens:
            if not parts:
                # An oversized first chunk must still make it into the prompt —
                # truncate it to the budget instead of emitting an empty context.
                header_tokens = len(_ENC.encode(header)) + 1
                budget = max_tokens - header_tokens
                if budget <= 0:
                    break
                content = _ENC.decode(_ENC.encode(block)[:budget])
                # Keep the fenced payload closed even after truncation.
                if FENCE_OPEN in content and FENCE_CLOSE not in content:
                    content += f"\n{FENCE_CLOSE}"
                parts.append(content)
                citations.append(
                    {
                        "source": i,
                        "doc_id": str(c["doc_id"]),
                        "title": meta.get("title"),
                        "page": meta.get("page"),
                    }
                )
            break
        parts.append(block)
        citations.append(
            {
                "source": i,
                "doc_id": str(c["doc_id"]),
                "title": meta.get("title"),
                "page": meta.get("page"),
            }
        )
        used += tokens
    return "\n\n---\n\n".join(parts), citations
