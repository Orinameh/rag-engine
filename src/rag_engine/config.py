from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiKeyEntry(BaseModel):
    """One API key: which workspace it belongs to and what it may do.

    ``role='admin'``  — may ingest documents into the workspace and query it.
    ``role='reader'`` — may query the workspace only.
    """

    workspace: str
    role: Literal["admin", "reader"] = "reader"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5435/rag"
    redis_url: str = "redis://localhost:6380/0"

    # Embeddings — quick-start defaults are small CPU-friendly models
    # (all-MiniLM-L6-v2, 384-d, ~90 MB → seconds to first query). For
    # production set EMBED_MODEL=BAAI/bge-m3 + EMBED_DIM=1024 (+ a matching
    # HALFVEC migration) and re-ingest. EMBED_DIM must match the model, and
    # models.py derives HALFVEC(dim) from this setting automatically.
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-change-me"
    llm_model: str = "gpt-4o-mini"

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 40
    rerank_top_n: int = 8
    rrf_k: int = 60

    max_context_tokens: int = 3000

    # Ingestion safety limits
    upload_max_bytes: int = 64_000_000  # 64 MB per uploaded file

    # PII protection: "off" | "llm_only" (mask before sending to the LLM) |
    # "all" (mask at ingest too — note: at-rest masking is irreversible).
    pii_masking: str = "llm_only"
    # PII mask style: "replace" ([EMAIL]) or "pseudonymize" (EMAIL_3f2a81c0).
    pii_mask_style: str = "replace"

    # Authentication & tenancy. JSON object mapping API key → workspace/role,
    # e.g. RAG_API_KEYS='{"sk-acme-admin": {"workspace": "acme", "role": "admin"},
    # "sk-acme-reader": {"workspace": "acme", "role": "reader"}}'.
    # Empty by default: with no keys configured, every protected endpoint
    # returns 401 until an operator creates at least one key.
    api_keys: dict[str, ApiKeyEntry] = Field(
        default={}, validation_alias=AliasChoices("RAG_API_KEYS", "API_KEYS")
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
