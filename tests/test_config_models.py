"""Model/dimension consistency between settings and schema."""

from rag_engine.config import Settings, get_settings
from rag_engine.models import Chunk


def test_embedding_column_dim_matches_settings():
    # The HALFVEC column is derived from EMBED_DIM; a fresh install with a
    # 384-d MiniLM gets halfvec(384), an m3 deployment gets halfvec(1024) —
    # but never a silent mismatch first seen at query time.
    dim = getattr(Chunk.__table__.c.embedding.type, "dim", None)
    assert str(dim) == str(get_settings().embed_dim)


def test_quick_start_profile_is_light():
    # Defaults, not get_settings(): the latter picks up local .env overrides
    # (e.g. a machine pinned to the bge-m3 production profile).
    s = Settings(_env_file=None)
    # Defaults must be small enough that first startup doesn't fetch GBs.
    assert "MiniLM" in s.embed_model or "small" in s.embed_model
    assert s.embed_dim <= 512


def test_docs_disabled_by_default(monkeypatch):
    """Unauthenticated /docs & /openapi.json must be off unless explicitly on."""
    monkeypatch.delenv("DOCS_ENABLED", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().docs_enabled is False
    finally:
        get_settings.cache_clear()
