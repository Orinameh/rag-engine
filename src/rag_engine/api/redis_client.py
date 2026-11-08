"""Shared async-Redis client factory for API-side state (limiter, ownership,
metrics queue gauge).

One place for the timeout policy, which is the fail-security of both guard
paths: redis-py's defaults (socket_timeout=None, socket_connect_timeout=None)
block indefinitely on a half-open connection, turning a Redis *hang* into an
API hang — exactly the outage the fail-open design promises to avoid. With
short sockets, every failure mode (refused, partitioned, hung) becomes a
RedisError/Timeout within ~2 s, which the call sites already catch and
degrade on.

Clients are cached per (consumer) so hot paths don't churn pools; ``close``
is never needed for the process-lifetime instances.
"""

from redis import asyncio as aioredis

CONNECT_TIMEOUT_SECONDS = 2
COMMAND_TIMEOUT_SECONDS = 2

_clients: dict[str, aioredis.Redis] = {}


def redis_client(name: str, url: str | None = None) -> aioredis.Redis:
    """A cached, timeout-hardened client. ``url`` defaults to settings."""
    if name in _clients:
        return _clients[name]
    if url is None:
        from rag_engine.config import get_settings

        url = get_settings().redis_url
    _clients[name] = aioredis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=CONNECT_TIMEOUT_SECONDS,
        socket_timeout=COMMAND_TIMEOUT_SECONDS,
    )
    return _clients[name]
