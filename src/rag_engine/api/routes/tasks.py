import logging

from celery.result import AsyncResult
from fastapi import APIRouter

from rag_engine.api.auth import CallerDep
from rag_engine.worker import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}")
async def task_status(task_id: str, caller: CallerDep) -> dict:
    res = AsyncResult(task_id, app=celery_app)
    if not res.ready():
        return {"task_id": task_id, "state": res.state, "result": None}
    if res.state == "FAILURE":
        # Celery's raw result on FAILURE is an exception object with a full
        # traceback; only surface a short message so internal details don't leak.
        logger.exception("Task %s failed", task_id, exc_info=res.result)
        message = str(res.result) or "task failed"
        return {
            "task_id": task_id,
            "state": res.state,
            "result": {"error": message[:500]},
        }
    return {"task_id": task_id, "state": res.state, "result": res.result}
