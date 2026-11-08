"""Rate-limit decision logic + fail-open behaviour (CI-safe: no Redis)."""

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError

from rag_engine.api import ratelimit
from rag_engine.config import get_settings


@pytest.fixture()
def limits(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS", '{"sk-a": {"workspace": "acme"}}')
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_fail_open_when_redis_down(limits, monkeypatch):
    """Redis outage must never take the API down (fail open by design)."""

    async def boom():
        raise RedisError("connection refused")

    class FakePipe:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def incr(self, _key):
            return self

        def expire(self, _key, _ttl):
            return self

        async def execute(self):
            await boom()

    class Fake:
        def pipeline(self, transaction=True):
            return FakePipe()

    monkeypatch.setattr(ratelimit, "redis_client", lambda *a, **k: Fake())
    # Must not raise, even though every window says "burst".
    await ratelimit.check_rate_limit("digest")


@pytest.mark.anyio
async def test_passes_under_limits(limits, monkeypatch):
    class FakePipe:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def incr(self, _key):
            return self

        def expire(self, _key, _ttl):
            return self

        async def execute(self):
            return [1, True, 1, True]  # both windows at 1/limit

    class Fake:
        def pipeline(self, transaction=True):
            return FakePipe()

    monkeypatch.setattr(ratelimit, "redis_client", lambda *a, **k: Fake())
    await ratelimit._count_and_check("digest")


@pytest.mark.anyio
async def test_429_over_minute(limits, monkeypatch):
    class FakePipe:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def incr(self, _key):
            return self

        def expire(self, _key, _ttl):
            return self

        async def execute(self):
            return [limits.rate_limit_per_minute + 1, True, 1, True]

    class Fake:
        def pipeline(self, transaction=True):
            return FakePipe()

    monkeypatch.setattr(ratelimit, "redis_client", lambda *a, **k: Fake())
    with pytest.raises(HTTPException) as err:
        await ratelimit._count_and_check("digest")
    assert err.value.status_code == 429
    headers = err.value.headers or {}
    assert "Retry-After" in headers


@pytest.mark.anyio
async def test_disabled_bypasses(limits, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    try:

        # With the limiter off even a comically high counter must not raise —
        # and we prove it by asserting the counting path is never consulted.
        async def should_not_be_called():
            raise AssertionError("counting must not run when disabled")

        monkeypatch.setattr(ratelimit, "_count_and_check", should_not_be_called)
        await ratelimit.check_rate_limit("digest")
    finally:
        get_settings.cache_clear()
