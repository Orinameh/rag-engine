"""Task-ownership guard logic (CI-safe: fake redis, no celery)."""

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError

from rag_engine.api import ownership


class FakeRedis:
    """In-memory subset of redis.asyncio (set/get with TTL ignored)."""

    def __init__(self):
        self.data: dict[str, str] = {}

    async def set(self, key, value, ex=None):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)


@pytest.fixture()
def fake(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(ownership, "redis_client", lambda *a, **k: r)
    return r


@pytest.mark.anyio
async def test_owner_can_poll(fake):
    await ownership.record_owner("t1", "acme")
    await ownership.verify_access("t1", "acme")  # must not raise


@pytest.mark.anyio
async def test_foreign_workspace_is_404(fake):
    await ownership.record_owner("t1", "acme")
    with pytest.raises(HTTPException) as err:
        await ownership.verify_access("t1", "globex")
    assert err.value.status_code == 404


@pytest.mark.anyio
async def test_unknown_task_is_404(fake):
    with pytest.raises(HTTPException) as err:
        await ownership.verify_access("never-existed", "acme")
    assert err.value.status_code == 404


@pytest.mark.anyio
async def test_redis_failure_fails_open(fake, monkeypatch):
    """Infrastructure failure ≠ access denial; the route stays usable."""
    # Not previously recorded AND Redis errors → policy: fail open.
    class Boom:
        async def get(self, _key):
            raise RedisError("connection lost")

    monkeypatch.setattr(ownership, "redis_client", lambda *a, **k: Boom())
    await ownership.verify_access("whatever", "acme")
