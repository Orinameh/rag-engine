from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag_engine.config import get_settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(get_settings().rerank_model, max_length=512)


def rerank(query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
    if not candidates:
        return []
    scores = get_reranker().predict(
        [(query, c["content"]) for c in candidates],
        batch_size=16,
        show_progress_bar=False,
    )
    ranked = sorted(
        zip(candidates, scores, strict=True), key=lambda x: float(x[1]), reverse=True
    )
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_n]]
