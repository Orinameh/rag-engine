import logging

from fastapi import APIRouter, Response

from rag_engine.api import metrics
from rag_engine.api.auth import AdminDep
from rag_engine.api.redis_client import redis_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["ops"])


@router.get("")
async def prometheus(caller: AdminDep) -> Response:
    """Prometheus exposition. Admin-only: counters expose internal capability."""
    queue_depth = None
    try:
        # Shared cached client (not one per scrape): scrapes re-occur every
        # few seconds and a throwaway pool per call leaks connections.
        queue_depth = await redis_client("metrics").llen("celery")
    except Exception as exc:  # noqa: BLE001 — metrics scraping must never fail
        logger.debug("queue depth unavailable: %s", exc)
    return Response(
        content=metrics.build_exposition(queue_depth=queue_depth),
        media_type="text/plain; version=0.0.4; charset=utf-8",
        status_code=200,
    )
