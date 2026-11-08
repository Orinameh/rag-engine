import asyncio
import logging
from collections.abc import AsyncGenerator  # ← AsyncGenerator, not AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.requests import Request

from rag_engine.api import metrics as mtr
from rag_engine.api.routes import (
    ingest,
    metrics_route,
    query,
    query_stream,
    tasks,
)
from rag_engine.config import get_settings
from rag_engine.db import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup / shutdown lifecycle. Runs once per worker process."""
    # 1. Fail fast on DB connectivity
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

    # 2. Force-create the LLM client and parse RAG_API_KEYS now: a malformed
    #    JSON body or a bad ApiKeyEntry must abort boot, not surface as a
    #    401/500 on the first authenticated request.
    from rag_engine.generation.llm import get_client

    get_settings().api_keys  # noqa: B018 — parse+validate; raises on bad config
    get_client()
    logger.info("settings parsed: %d API key(s) configured", len(get_settings().api_keys))

    # Structured JSON logging for the running process (uvicorn's handlers are
    # configured by now, so substitution is final).
    from rag_engine.logging_setup import configure_structured_logging

    configure_structured_logging()

    # 3. Preload models so the first query isn't a 60s cold start
    from rag_engine.retrieval.embedder import get_embedder
    from rag_engine.retrieval.reranker import get_reranker

    await asyncio.to_thread(get_embedder)
    await asyncio.to_thread(get_reranker)

    yield

    # 4. Graceful shutdown
    await engine.dispose()


async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Profile every HTTP request into the in-process counter store.

    Fail-never: a recording error must not affect the response at all.
    """
    start = perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        try:
            # Cardinality guard: only routes the app actually serves get their
            # own series. Unmatched paths (404 scans, probes) collapse into the
            # single "{unmatched}" bucket — middlewares see arbitrary request
            # URLs, and an unbounded key would let path-fuzzing grow memory
            # forever. Matched routes use Starlette's already-templated
            # route.path (e.g. /tasks/{task_id}); uuid-collapse is the fallback
            # (tests + defensive) for any route that embeds dynamic ids.
            route = request.scope.get("route")
            if route is not None and getattr(route, "path", None):
                path = route.path
            else:
                path = "{unmatched}"
            mtr.record(request.method, path, status, perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 — profiling must never break traffic
            logger.debug("metrics recording failed: %s", exc)


app = FastAPI(title="RAG Engine", version="0.1.0", lifespan=lifespan)

app.middleware("http")(metrics_middleware)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(query_stream.router)
app.include_router(tasks.router)
app.include_router(metrics_route.router)


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
