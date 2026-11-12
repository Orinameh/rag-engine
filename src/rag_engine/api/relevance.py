"""Relevance gate: refuse queries whose best rerank score says "no match".

Retrieval is similarity-based — it ALWAYS returns the nearest chunks, even
for a question with nothing to do with the corpus. Without this gate the LLM
then decides whether to answer or say "I don't know", which depends entirely
on how well the (small) generator follows instructions. The gate moves that
decision to a number we can calibrate: bge-reranker sigmoid scores separate
the cases cleanly (on-corpus ~0.98+, off-corpus ~0.0 — see
scripts/calibrate_gate.py), so a top-1 below the threshold refuses the query
BEFORE any generation: no hallucination path, no token cost.

Two knobs (config.py):
- ``rerank_min_score`` — the query gate on the top-1 score; 0 disables.
- ``rerank_chunk_min_score`` — per-chunk floor; chunks below it are trimmed
  from the context even when the query passes, so near-noise tail chunks
  don't ride along. 0 disables (pure top-N ranking as before).
"""

from rag_engine.config import get_settings

REFUSAL_MESSAGE = (
    "no relevant material found in this workspace for that question"
)


def apply_relevance_gate(top: list[dict]) -> tuple[list[dict], str | None]:
    """Return (chunks to use, refusal message) for a reranked candidate list.

    Empty input refuses too: an empty retrieval previously fed the LLM an
    empty context and hoped for the best; refusing is strictly better.
    """
    s = get_settings()
    if not top:
        return [], REFUSAL_MESSAGE
    if s.rerank_min_score > 0 and float(top[0]["rerank_score"]) < s.rerank_min_score:
        return [], REFUSAL_MESSAGE
    floor = s.rerank_chunk_min_score
    if floor > 0:
        kept = [c for c in top if float(c["rerank_score"]) >= floor]
        if kept:  # the gate passed — never trim away the last relevant chunk
            return kept, None
    return top, None
