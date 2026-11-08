"""Task-ownership guard for /tasks/{task_id} polling.

Tasks created by workspace A's admin must be pollable only by workspace A
callers. Celery itself has no tenancy, so dispatch records
``task_owner:<task_id> → workspace`` in Redis; the tasks route rejects (with
404 — no existence leak) anything voted another workspace's task.

TTL: 7 days — long past any realistic poll window, cheap to keep.
Fail-closed on *unknown* ownership: a task with no recorded owner (pre-upgrade
tasks, stale Redis) is 404, because guessing becomes the only channel.
Fail-open only on infrastructure errors reaching Redis (the tasks route must
stay available during a Celery outage).
"""

import logging

from fastapi import HTTPException
from redis.exceptions import RedisError

from rag_engine.api.redis_client import redis_client

logger = logging.getLogger(__name__)

OWNER_TTL_SECONDS = 7 * 24 * 3600


async def record_owner(task_id: str, workspace: str) -> None:
    """Called by /ingest dispatch right after the task is enqueued."""
    try:
        await redis_client("ownership").set(
            f"task_owner:{task_id}", workspace, ex=OWNER_TTL_SECONDS
        )
    except RedisError as exc:
        # The task still runs; polling just stays fail-closed (404) for it.
        logger.warning("could not record task owner for %s: %s", task_id, exc)


async def verify_access(task_id: str, workspace: str) -> None:
    """404s tasks the caller's workspace doesn't own (or never owned)."""
    try:
        owner = await redis_client("ownership").get(f"task_owner:{task_id}")
    except RedisError as exc:
        logger.warning("task-ownership check degraded for %s: %s", task_id, exc)
        return  # fail open only on infrastructure failure
    if owner is None or owner != workspace:
        logger.warning(
            "task %s polled by workspace %r (owner: %r)",
            task_id,
            workspace,
            owner,
        )
        raise HTTPException(status_code=404, detail="Not Found")
