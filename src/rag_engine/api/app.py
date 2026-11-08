import asyncio
import logging
from collections.abc import AsyncGenerator  # ← AsyncGenerator, not AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.requests import Request

from rag_engine.api.routes import ingest, query, query_stream, tasks
from rag_engine.config import get_settings
from rag_engine.db import engine

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup / shutdown lifecycle. Runs once per worker process."""
    # 1. Fail fast on DB connectivity
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

    # 2. Preload models so the first query isn't a 60s cold start
    from rag_engine.retrieval.embedder import get_embedder
    from rag_engine.retrieval.reranker import get_reranker

    await asyncio.to_thread(get_embedder)
    await asyncio.to_thread(get_reranker)

    yield

    # 3. Graceful shutdown
    await engine.dispose()


app = FastAPI(title="RAG Engine", version="0.1.0", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(query_stream.router)
app.include_router(tasks.router)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log everything, leak nothing.

    HTTPException / RequestValidationError keep their FastAPI-generated
    bodies (415/413/401/403/422/...); this only catches what would otherwise
    escape as a raw 500 with internals in it.
    """
    logger.exception("Unhandled error serving %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
async def health() -> JSONResponse:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — health must never raise; report the cause
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": str(exc)[:200]},
        )
    return JSONResponse(content={"status": "ok", "model": get_settings().embed_model})
