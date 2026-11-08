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
            # One element per retrieved chunk: a chunk matching on either its
            # doc UUID or its source_uri counts once, so appending both
            # identifiers would inflate MRR/NDCG ranks.
            ident = doc_id
            if source and doc_id not in relevant and source in relevant:
                ident = source
            retrieved.append(ident)

        rows.append(
            {
                "question": case["question"],
                "recall": recall_at_k(retrieved, relevant, k),
                "mrr": mrr(retrieved, relevant),
                "ndcg": ndcg_at_k(retrieved, relevant, k),
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
