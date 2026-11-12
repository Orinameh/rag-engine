#!/usr/bin/env python3
"""One-off calibration probe for RERANK_MIN_SCORE (see README config).

Scores every (question, chunk) pair in each workspace with the live
cross-encoder and prints the top score per question — on-corpus questions
(should score high) vs off-corpus/adjacent ones (should score low). The
threshold sits in the gap.
"""

import asyncio

from sqlalchemy import select

from rag_engine.db import SessionFactory
from rag_engine.models import Chunk, Document
from rag_engine.retrieval.reranker import get_reranker

ACME = [
    "What is the maximum operating temperature before contacting ops?",
    "What is the deployment rollback window after a failed release?",
    "How long do I boil an egg for?",
    "What is the capital of France?",
    "Who won the 2019 Champions League final?",
    "How do I reset my Netflix password?",
]
GLOBEX = [
    "What is the springfield calibration phrase?",
    "What packaging standard applies to board-level electronics?",
    "How long do I boil an egg for?",
    "What is the capital of France?",
]


async def main() -> None:
    async with SessionFactory() as session:
        rows = (
            await session.execute(
                select(Chunk.content, Document.workspace, Document.title)
                .join(Document, Chunk.doc_id == Document.id)
            )
        ).all()

    by_ws: dict[str, list[tuple[str, str]]] = {}
    for content, workspace, title in rows:
        by_ws.setdefault(workspace, []).append((content, title))

    rr = get_reranker()
    for ws, questions in (("acme", ACME), ("globex", GLOBEX)):
        chunks = by_ws.get(ws, [])
        if not chunks:
            continue
        print(f"== {ws} ({len(chunks)} chunks) ==")
        for q in questions:
            scores = rr.predict([(q, c) for c, _ in chunks], show_progress_bar=False)
            top = max(float(s) for s in scores)
            title = chunks[int(scores.argmax())][1] if hasattr(scores, "argmax") else ""
            print(f"  top={top:.3f}  {q[:58]}  ({title})")


if __name__ == "__main__":
    asyncio.run(main())
