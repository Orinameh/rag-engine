"""Simple API-key authentication + workspace-scoped authorization.

One caller = one API key = one workspace + one role:

- ``admin``  — may ingest documents into the workspace and query it.
- ``reader`` — may query the workspace only.

Keys live in settings (``RAG_API_KEYS`` JSON). There is deliberately no
database of users — for a small fleet the key store is this file; replacing
``_lookup_key`` with a DB-backed implementation later keeps every other layer
untouched. In memory the keys exist only as SHA-256 digests (config.py), so
reprs, tracebacks, and dumps never contain usable credentials, and resolution
is a constant-time digest comparison.

Tenancy is enforced server-side in the retrieval SQL: every query is scoped to
the caller's workspace before any expensive compute runs, and client-supplied
``metadata_filter`` can only narrow, never widen that scope.
"""

from hashlib import sha256
from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

from rag_engine.api.ratelimit import INVALID_AUTH_BUCKET, check_rate_limit
from rag_engine.config import ApiKeyEntry, get_settings

# RFC 6750: a 401 from a Bearer-protected resource should carry this header —
# it tells standards-compliant clients (and humans) which scheme was expected.
_WWW_AUTH = {"WWW-Authenticate": 'Bearer realm="rag"'}


class Caller(BaseModel):
    """The authenticated principal for one request.

    ``key_id`` is the SHA-256 digest of the API key — an identity suitable for
    rate-limit keys and logs without carrying the plaintext secret.
    """

    workspace: str
    role: str
    key_id: str


def key_digest(token: str) -> str:
    """What the config stores for a given plaintext key."""
    return sha256(token.encode()).hexdigest()


def _lookup_key(token: str) -> ApiKeyEntry | None:
    """Resolve an API key to its workspace/role. Swap this for a DB later.

    The configured map is keyed by SHA-256 digests (see config.py), and each
    comparison is constant-time: an attacker probing many bad tokens learns
    nothing from response timing, and the digests alone are not credentials.
    """
    digest = key_digest(token)
    for stored, entry in get_settings().api_keys.items():
        if compare_digest(stored, digest):
            return entry
    return None


def authenticate(token: str) -> Caller:
    entry = _lookup_key(token)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unknown API key",
            headers=_WWW_AUTH,
        )
    return Caller(
        workspace=entry.workspace, role=entry.role, key_id=key_digest(token)
    )


async def require_caller(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Caller:
    """FastAPI dependency: Bearer token or X-API-Key header → Caller.

    Also the single choke point for per-key rate limits: every authenticated
    surface goes through this dependency (see api/ratelimit.py)."""
    token = None
    if authorization:
        scheme, _, value = authorization.partition(" ")
        token = value.strip() if scheme.lower() == "bearer" else None
        if not token:
            await check_rate_limit(INVALID_AUTH_BUCKET)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header must be 'Bearer <api-key>'",
                headers=_WWW_AUTH,
            )
    if token is None:
        token = x_api_key
    if not token:
        await check_rate_limit(INVALID_AUTH_BUCKET)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials; pass 'Authorization: Bearer <api-key>' or 'X-API-Key'",
            headers=_WWW_AUTH,
        )
    try:
        caller = authenticate(token)
    except HTTPException:
        # Failed auth gets no key digest, so it draws against ONE shared
        # fixed-rate bucket (ratelimit.INVALID_AUTH_BUCKET): unlimited
        # key-probing and 401 floods are not free, and a 429 also tells a
        # legitimate caller-with-a-typo to back off. Redis failure fail-opens
        # inside check_rate_limit, so auth stays usable in a broker outage.
        await check_rate_limit(INVALID_AUTH_BUCKET)
        raise
    await check_rate_limit(caller.key_id)
    return caller


async def require_admin(caller: Annotated[Caller, Depends(require_caller)]) -> Caller:
    """FastAPI dependency: same as require_caller but role must be admin."""
    if caller.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires the 'admin' role",
        )
    return caller


def scoped_metadata_filter(user_filter: dict | None, workspace: str) -> dict:
    """Merge the client's filter with the mandatory workspace scope.

    The server's workspace value always wins, so a caller can narrow its own
    slice (authority=, category=, ...) but can never reach another workspace's
    chunks, no matter what keys it puts in ``metadata_filter``.
    """
    out = dict(user_filter or {})
    out["workspace"] = workspace
    return out


CallerDep = Annotated[Caller, Depends(require_caller)]
AdminDep = Annotated[Caller, Depends(require_admin)]
