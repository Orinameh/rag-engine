import json
from functools import lru_cache
from hashlib import sha256
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr
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

    # Dev-only defaults. They mirror docker-compose.yaml so `docker compose up`
    # + `uv run uvicorn …` works with zero configuration. OS env vars and
    # .env always win; NOTHING in a production deployment reads these.
    # Production must set DATABASE_URL explicitly (a real password, a real
    # host) — the default credentials only exist for the local containers.
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
    # SecretStr: the key renders as ********** whenever Settings is repr'd or
    # logged (e.g. inside exception messages) instead of leaking in plaintext.
    # Empty default means exactly that: no key. Local servers (Ollama, LM
    # Studio) ignore it entirely; get_generation_client() fails fast with a
    # clear error if a PUBLIC provider is selected without one.
    llm_api_key: SecretStr = SecretStr("")
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

    # Per-key rate limiting (api/ratelimit.py). Both counters are fixed
    # Redis windows: no caller may exceed either. Set 0 to disable a window
    # individually; rate_limit_enabled=False bypasses the limiter entirely
    # (Redis failures always fail open regardless of this setting).
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1200

    # Relevance gate (api/relevance.py): if the top-1 rerank score is below
    # rerank_min_score, the query is refused ("no relevant material") BEFORE
    # the LLM runs — no hallucination path, no token cost. Calibrated on the
    # live profile (scripts/calibrate_gate.py): on-corpus ~0.98+, off-corpus
    # ~0.0; 0.50 sits mid-gap. 0 disables the gate.
    rerank_min_score: float = 0.50
    # Per-chunk floor: chunks under this score are trimmed from the context
    # even when the query passes the gate (tail-noise guard). 0 = keep top-N.
    rerank_chunk_min_score: float = 0.10

    # Unauthenticated /docs, /redoc, /openapi.json are off by default
    # (app.py). Enable only for development.
    docs_enabled: bool = False

    # Authentication & tenancy. RAG_API_KEYS is a JSON object mapping key →
    # workspace/role, e.g.
    #   RAG_API_KEYS='{"sk-acme-admin": {"workspace": "acme", "role": "admin"},
    #                  "sk-acme-reader": {"workspace": "acme", "role": "reader"}}'
    #
    # SECURITY — two deliberate properties:
    # 1. It is declared as a SCALAR string, not a dict field. pydantic-settings
    #    deep-merges dict-valued fields across sources, which would silently
    #    RESURRECT revoked keys from a stale .env whenever a process-env
    #    override was present in one source and the .env in another. A string
    #    fully replaces: process env wins over .env, no merge, revocations
    #    take effect.
    # 2. Plaintext keys never live in the model. Access goes through the
    #    `api_keys` property, which parses and hashes on demand and returns
    #    {SHA-256(key) → entry}. The hashed map appears in no repr, no
    #    traceback, no memory dump; digests alone are not credentials (see
    #    api/auth.py for constant-time resolution).
    rag_api_keys_json: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_API_KEYS", "API_KEYS"),
        repr=False,
    )

    @property
    def api_keys(self) -> dict[str, ApiKeyEntry]:
        """SHA-256(key) → entry. Blank keys are dropped, never wildcards.

        Parsed results are memoized per distinct raw value: ``_lookup_key``
        calls this on every authenticated request, and re-parsing JSON +
        re-hashing each key per request is wasted CPU. Memoizing on the raw
        string is the safe point of recall — a config change (revocation)
        REPLACES this field's value (scalar env override, see above), which
        changes the raw string and forces a re-parse. Same raw string ⇒ same
        map is a tautology.
        """
        raw = (self.rag_api_keys_json or "").strip()
        if not raw:
            return {}
        cached = _api_keys_cache.get(raw)
        if cached is None:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise TypeError("RAG_API_KEYS must be a JSON object")
            cached = {
                sha256(key.encode()).hexdigest(): ApiKeyEntry.model_validate(entry)
                for key, entry in data.items()
                if key
            }
            _api_keys_cache[raw] = cached
        return cached


@lru_cache
def get_settings() -> Settings:
    # Single lru-cached instance; .env loading comes from model_config.
    return Settings()


# Memoization for Settings.api_keys, keyed by the raw JSON string (see the
# property's docstring for why that invariant makes this safe).
_api_keys_cache: dict[str, dict[str, ApiKeyEntry]] = {}
