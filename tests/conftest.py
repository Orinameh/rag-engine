"""Shared test-environment guards.

The per-key rate limiter talks to Redis; the unit suite must stay
deterministic without infrastructure, so the limiter is DISABLED by default
here (fail-fast branch in ratelimit.check_rate_limit). tests that exercise
the limiter's logic set their own env within the test and clear the cache.
"""

import pytest

from rag_engine.config import get_settings


@pytest.fixture(autouse=True)
def _rate_limit_off(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
