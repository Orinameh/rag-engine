import asyncio
import hashlib
from functools import lru_cache

import orjson
import redis.asyncio as aioredis
from sentence_transformers import SentenceTransformer

from rag_engine.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embed_model, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vecs = model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )
    return [v.tolist() for v in vecs]


_redis: aioredis.Redis | None = None


def _redis_client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(get_settings().redis_url)
    return _redis


async def embed_query_cached(text: str) -> list[float]:
    """Query-side embeddings cached by content hash in Redis.

    The embedding model runs on CPU and is blocking; it must run in a worker
    thread so the API's event loop stays responsive while it sits on cache misses.
    """
    r = _redis_client()
    key = f"qemb:{hashlib.sha256(text.encode()).hexdigest()}"
    if hit := await r.get(key):
        return orjson.loads(hit)
    vec = (await asyncio.to_thread(embed_texts, [text]))[0]
    await r.set(key, orjson.dumps(vec), ex=3600)
    return vec
