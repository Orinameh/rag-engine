"""Simple API-key authentication + workspace-scoped authorization.

One caller = one API key = one workspace + one role:

- ``admin``  — may ingest documents into the workspace and query it.
- ``reader`` — may query the workspace only.

Keys live in settings (``RAG_API_KEYS`` JSON). There is deliberately no
database of users — for a small fleet the key store is this file; replacing
``_lookup_key`` with a DB-backed implementation later keeps every other layer
untouched.

Tenancy is enforced server-side in the retrieval SQL: every query is scoped to
the caller's workspace before any expensive compute runs, and client-supplied
``metadata_filter`` can only narrow, never widen that scope.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

from rag_engine.config import ApiKeyEntry, get_settings


class Caller(BaseModel):
    """The authenticated principal for one request."""

    workspace: str
    role: str


def _lookup_key(token: str) -> ApiKeyEntry | None:
    """Resolve an API key to its workspace/role. Swap this for a DB later."""
    return get_settings().api_keys.get(token)


def authenticate(token: str) -> Caller:
    entry = _lookup_key(token)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unknown API key",
        )
    return Caller(workspace=entry.workspace, role=entry.role)


async def require_caller(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Caller:
    """FastAPI dependency: Bearer token or X-API-Key header → Caller."""
    token = None
    if authorization:
        scheme, _, value = authorization.partition(" ")
        token = value.strip() if scheme.lower() == "bearer" else None
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header must be 'Bearer <api-key>'",
            )
    if token is None:
        token = x_api_key
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials; pass 'Authorization: Bearer <api-key>' or 'X-API-Key'",
        )
    return authenticate(token)


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
