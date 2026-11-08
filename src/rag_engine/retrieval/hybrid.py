import orjson
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.config import get_settings

_RRF_K = get_settings().rrf_k


async def hybrid_search(
    session: AsyncSession,
    query: str,
    query_vec: list[float],
    top_k: int = 40,
    metadata_filter: dict | None = None,
) -> list[dict]:
    filter_clause = (
        "AND metadata @> CAST(:metadata_filter AS jsonb)" if metadata_filter else ""
    )
    params = {
        "q": query,
        "vec": str(query_vec),
        "k": top_k,
        "metadata_filter": orjson.dumps(metadata_filter).decode()
        if metadata_filter
        else None,
    }

    vector_sql = text(
        f"""
        SELECT id, doc_id, content, metadata,
               1 - (embedding <=> CAST(:vec AS halfvec)) AS score
        FROM chunks
        WHERE embedding IS NOT NULL {filter_clause}
        ORDER BY embedding <=> CAST(:vec AS halfvec)
        LIMIT :k
        """
    )
    lexical_sql = text(
        f"""
        SELECT id, doc_id, content, metadata,
               ts_rank_cd(tsv, plainto_tsquery('english', :q)) AS score
        FROM chunks
        WHERE tsv @@ plainto_tsquery('english', :q) {filter_clause}
        ORDER BY score DESC
        LIMIT :k
        """
    )

    vec_rows = (await session.execute(vector_sql, params)).mappings().all()
    lex_rows = (await session.execute(lexical_sql, params)).mappings().all()

    scores: dict[int, float] = {}
    for rank, row in enumerate(vec_rows, 1):
        scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (_RRF_K + rank)
    for rank, row in enumerate(lex_rows, 1):
        scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (_RRF_K + rank)

    fused = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    by_id = {r["id"]: dict(r) for r in [*vec_rows, *lex_rows]}
    return [by_id[i] for i in fused if i in by_id]
