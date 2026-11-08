import asyncio

import openai
from fastapi import APIRouter, HTTPException
from openai import APIError
from pydantic import BaseModel, Field
from sqlalchemy import select

from rag_engine.api.auth import CallerDep, scoped_metadata_filter
from rag_engine.api.deps import SessionDep
from rag_engine.config import get_settings
from rag_engine.generation.llm import generate_answer
from rag_engine.models import Document
from rag_engine.retrieval.context import build_context
from rag_engine.retrieval.embedder import embed_query_cached
from rag_engine.retrieval.hybrid import hybrid_search
from rag_engine.retrieval.reranker import rerank

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    metadata_filter: dict | None = None
    top_k: int = Field(default=40, ge=1, le=100)


class Citation(BaseModel):
    source: int
    doc_id: str
    title: str | None
    page: int | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved: int
    reranked: int


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest, caller: CallerDep, session: SessionDep
) -> QueryResponse:
    s = get_settings()
    query_vec = await embed_query_cached(body.question)

    candidates = await hybrid_search(
        session,
        body.question,
        query_vec,
        top_k=body.top_k,
        # Server-side tenancy: the workspace key always wins, so client
        # filters can only narrow the caller's own slice.
        metadata_filter=scoped_metadata_filter(body.metadata_filter, caller.workspace),
    )
    top = await asyncio.to_thread(rerank, body.question, candidates, s.rerank_top_n)

    doc_ids = {c["doc_id"] for c in top}
    docs = {
        d.id: d.title
        for d in (
            await session.execute(select(Document).where(Document.id.in_(doc_ids)))
        ).scalars()
    }
    for c in top:
        c.setdefault("metadata", {})
        c["metadata"]["title"] = docs.get(c["doc_id"])

    context, citations = build_context(top, max_tokens=s.max_context_tokens)
    try:
        answer = await generate_answer(body.question, context)
    except APIError as exc:
        # Provider-side failure (bad key, quota, 5xx) — not the caller's fault.
        # 502 keeps the global 500 handler reserved for genuine server bugs.
        status = 503 if isinstance(exc, openai.RateLimitError) else 502
        raise HTTPException(status_code=status, detail="generation failed") from exc

    return QueryResponse(
        answer=answer,
        citations=[Citation(**c) for c in citations],
        retrieved=len(candidates),
        reranked=len(top),
    )
