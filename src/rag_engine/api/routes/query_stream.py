import asyncio
import logging

import orjson
from fastapi import APIRouter
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from rag_engine.api.auth import CallerDep, scoped_metadata_filter
from rag_engine.api.relevance import apply_relevance_gate
from rag_engine.api.routes.query import QueryRequest
from rag_engine.config import get_settings
from rag_engine.db import SessionFactory
from rag_engine.generation.llm import generate_answer_stream
from rag_engine.models import Document
from rag_engine.retrieval.context import build_context
from rag_engine.retrieval.embedder import embed_query_cached
from rag_engine.retrieval.hybrid import hybrid_search
from rag_engine.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/stream")
async def query_stream(body: QueryRequest, caller: CallerDep) -> EventSourceResponse:
    s = get_settings()

    # Phase 1: retrieval + rerank in a short-lived DB session
    query_vec = await embed_query_cached(body.question)
    async with SessionFactory() as session:
        candidates = await hybrid_search(
            session,
            body.question,
            query_vec,
            top_k=body.top_k,
            # Server-side tenancy, same as /query: clients can only narrow.
            metadata_filter=scoped_metadata_filter(
                body.metadata_filter, caller.workspace
            ),
        )
        top = await asyncio.to_thread(rerank, body.question, candidates, s.rerank_top_n)

        doc_ids = {c["doc_id"] for c in top}
        docs = {
            d.id: d.title
            for d in (
                await session.execute(select(Document).where(Document.id.in_(doc_ids)))
            ).scalars()
        }
    # Session closed — no DB connection is held during streaming.

    for c in top:
        c.setdefault("metadata", {})
        c["metadata"]["title"] = docs.get(c["doc_id"])

    # Relevance gate: an SSE refusal is an explicit `error` event followed by
    # close — machine-readable, and the generator never runs (see
    # api/relevance.py; /query answers 200 with the same message).
    top, refusal = apply_relevance_gate(top)
    # build_context([]) yields ("", []) — keeps context a str on either path.
    context, citations = build_context(top, max_tokens=s.max_context_tokens)

    # Phase 2: stream tokens (no DB)
    async def event_gen():
        if refusal is not None:
            yield {"event": "error", "data": refusal}
            return
        yield {"event": "citations", "data": orjson.dumps(citations).decode()}
        try:
            async for token in generate_answer_stream(body.question, context):
                yield {"event": "token", "data": token}
        except Exception:
            # Without an explicit error event a client would hang forever when
            # the LLM dies mid-stream.
            logger.exception("LLM stream failed")
            yield {"event": "error", "data": "generation failed"}
            return
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_gen(), ping=15)
