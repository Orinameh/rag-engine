import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.evaluation.metrics import mrr, ndcg_at_k, recall_at_k
from rag_engine.retrieval.embedder import embed_texts
from rag_engine.retrieval.hybrid import hybrid_search
from rag_engine.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


async def run_eval(session: AsyncSession, dataset_path: str, k: int = 10) -> dict:
    cases = json.loads(Path(dataset_path).read_text())

    # Match ground truth by stable identifiers, not by ingest-time UUIDs:
    # datasets rebuild across environments, but titles stay fixed (the seed
    # script sets deterministic doc titles). Only documents whose titles
    # actually appear in the dataset are fetched — a full scan of every
    # document's (id, title) would pull O(#docs) memory into eval.
    from sqlalchemy import select

    from rag_engine.models import Document

    wanted = {str(i) for case in cases for i in case["relevant"]}
    titles: dict[str, str] = {
        str(doc_id): title
        for doc_id, title in (
            await session.execute(
                select(Document.id, Document.title).where(
                    Document.title.in_(wanted)
                )
            )
        ).all()
    }

    rows = []
    for case in cases:
        relevant = set(case["relevant"])
        qvec = await asyncio.to_thread(embed_texts, [case["question"]])
        candidates = await hybrid_search(session, case["question"], qvec[0], top_k=40)
        reranked = await asyncio.to_thread(rerank, case["question"], candidates, k)

        retrieved: list[str] = []
        for c in reranked:
            doc_id = str(c["doc_id"])
            meta = c.get("metadata") or {}
            source = str(meta["source"]) if meta.get("source") else None
            title = titles.get(doc_id)
            # One element per retrieved chunk: a chunk matching on doc UUID,
            # source_uri, or title counts once, so appending multiple
            # identifiers for the same chunk would inflate MRR/NDCG ranks.
            ident = doc_id
            if source and doc_id not in relevant and source in relevant:
                ident = source
            elif title and doc_id not in relevant and title in relevant:
                ident = title
            retrieved.append(ident)

        # Binary relevance here is DOC-level, but the pipeline is chunk-level:
        # several chunks of the same doc all map to one identifier. Dedupe to
        # each identifier's first (best) rank — otherwise a doc split into many
        # chunks would credit DCG repeatedly while IDCG counts it once, pushing
        # NDCG above 1.0 (bounded-metric violation).
        seen: set[str] = set()
        ranked_docs = []
        for ident in retrieved:
            if ident not in seen:
                seen.add(ident)
                ranked_docs.append(ident)

        rows.append(
            {
                "question": case["question"],
                "recall": recall_at_k(ranked_docs, relevant, k),
                "mrr": mrr(ranked_docs, relevant),
                "ndcg": ndcg_at_k(ranked_docs, relevant, k),
            }
        )

    n = len(rows) or 1
    return {
        "n": len(rows),
        f"recall@{k}": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        f"ndcg@{k}": sum(r["ndcg"] for r in rows) / n,
        "per_case": rows,
    }
