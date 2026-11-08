"""Per-key rate limiting on Redis fixed-window counters.

Every authenticated request increments two windows for the caller's key
digest: one per-minute (burst protection) and one per-hour (sustained
abuse). Exceeding either raises 429 with a Retry-After hint.

Design notes:
- Fail-open: if Redis is unavailable (the same outage breaks Celery), requests
  pass through rather than turning one incident into a full outage. The
  limiter is abuse protection, not a correctness boundary. Clients carry
  short socket timeouts (api/redis_client.py) so a *hung* Redis degrades to
  an error → fail open, instead of blocking requests the way the
  redis-py defaults would.
- The counter key is the SHA-256 of the caller's API key (never the plaintext
  token), so rate-limit state can't reveal credentials either.
- INCR+EXPIRE run in one transaction and are atomic per caller: two
  concurrent requests can't both see "59 of 60".
- Scope: the limiter runs AFTER a key authenticates, so it throttles
  authenticated callers. Failed auth attempts share a fixed-rate bucket (see
  auth.py) — an invalid key has no digest of its own, and unlimited
  key-probing must not ride for free.
- Fixed-window known trade-off: up to ~2x the named limit can pass across a
  window boundary (tail of one window + head of the next). Acceptable for an
  abuse guard; a sliding-window/sliding-log design removes it if ever needed.
"""

import logging
import time

from fastapi import HTTPException
from redis.exceptions import RedisError

from rag_engine.api.redis_client import redis_client
from rag_engine.config import get_settings

logger = logging.getLogger(__name__)


async def check_rate_limit(key_digest: str) -> None:
    """Raise 429 when the caller exceeds per-minute or per-hour limits."""
    s = get_settings()
    if not s.rate_limit_enabled:
        return
    if s.rate_limit_per_minute <= 0 and s.rate_limit_per_hour <= 0:
        return
    try:
        await _count_and_check(key_digest)
    except RedisError as exc:
        # Fail open: limiter must not become a second outage vector.
        logger.warning("rate limiter unavailable, failing open: %s", exc)


# Shared identity for unauthenticated attempts inside check_rate_limit: they
# have no key digest, so they all draw against ONE fixed-rate bucket instead
# of an unlimited free pass. Legit traffic never lands here; an attacker
# probing keys is capped at the same 60/min / 1200/hr ceilings — slow enough
# to make brute-force pointless, and a flood of 401s stops being free or a
# trivial amplifier.
INVALID_AUTH_BUCKET = "invalid-auth"


async def _count(key_digest: str) -> tuple[int, int]:
    """INCR both fixed windows for one identity. Returns (minute, hour) counts.

    INCR+EXPIRE run in one transaction, so two concurrent requests can't both
    see "59 of 60".
    """
    now = int(time.time())
    minute_key = f"ratelimit:{key_digest}:m{now // 60}"
    hour_key = f"ratelimit:{key_digest}:h{now // 3600}"
    async with redis_client("ratelimit").pipeline(transaction=True) as pipe:
        pipe.incr(minute_key)
        pipe.expire(minute_key, 70)
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3700)
        counts = await pipe.execute()
    return counts[0], counts[2]


async def _count_and_check(key_digest: str) -> None:
    s = get_settings()
    now = int(time.time())
    m_count, h_count = await _count(key_digest)

    if s.rate_limit_per_minute > 0 and m_count > s.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {s.rate_limit_per_minute} requests/minute "
                "for this API key"
            ),
            headers={"Retry-After": str(60 - now % 60)},
        )
    if s.rate_limit_per_hour > 0 and h_count > s.rate_limit_per_hour:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {s.rate_limit_per_hour} requests/hour "
                "for this API key"
            ),
            headers={"Retry-After": str(3600 - now % 3600)},
        )
