"""Failed-auth requests must draw the shared invalid-auth bucket, not ride
for free: the limiter normally runs only after a key authenticates."""

import pytest
from fastapi import HTTPException

from rag_engine.api import auth as auth_mod
from rag_engine.api import ratelimit as ratelimit_mod
from rag_engine.config import get_settings


@pytest.fixture()
def _keys_env(monkeypatch):
    monkeypatch.setenv(
        "RAG_API_KEYS",
        '{"sk-a": {"workspace": "acme", "role": "reader"}}',
    )
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_unknown_key_counts_against_invalid_bucket(monkeypatch):
    seen: list[str] = []

    async def spy(key: str) -> None:
        seen.append(key)

    monkeypatch.setattr(auth_mod, "check_rate_limit", spy)
    with pytest.raises(HTTPException) as err:
        await auth_mod.require_caller(
            authorization="Bearer sk-not-a-real-key", x_api_key=None
        )
    assert err.value.status_code == 401
    assert seen == [ratelimit_mod.INVALID_AUTH_BUCKET]
    # RFC 6750: 401s from a Bearer resource advertise the scheme.
    headers = err.value.headers or {}
    assert headers["WWW-Authenticate"] == 'Bearer realm="rag"'


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_missing_credentials_count_against_invalid_bucket(monkeypatch):
    seen: list[str] = []

    async def spy(key: str) -> None:
        seen.append(key)

    monkeypatch.setattr(auth_mod, "check_rate_limit", spy)
    with pytest.raises(HTTPException) as err:
        await auth_mod.require_caller(authorization=None, x_api_key=None)
    assert err.value.status_code == 401
    assert seen == [ratelimit_mod.INVALID_AUTH_BUCKET]
    assert (err.value.headers or {})["WWW-Authenticate"] == 'Bearer realm="rag"'


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_valid_key_is_limited_by_its_own_digest(monkeypatch):
    from hashlib import sha256

    seen: list[str] = []

    async def spy(key: str) -> None:
        seen.append(key)

    monkeypatch.setattr(auth_mod, "check_rate_limit", spy)
    caller = await auth_mod.require_caller(
        authorization=None, x_api_key="sk-a"
    )
    expected = sha256(b"sk-a").hexdigest()
    assert seen == [expected]
    assert caller.workspace == "acme"
