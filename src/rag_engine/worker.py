import asyncio
import hashlib
import logging
from pathlib import Path
from typing import cast

import tiktoken
from celery import Celery
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_engine.config import get_settings
from rag_engine.ingestion.chunkers import RecursiveChunker
from rag_engine.ingestion.parsers import parse_document
from rag_engine.models import Chunk, Document
from rag_engine.retrieval.embedder import embed_texts
from rag_engine.retrieval.pii import mask_pii

logger = logging.getLogger(__name__)

settings = get_settings()
_ENC = tiktoken.get_encoding("cl100k_base")
EMBED_BATCH = 64

# Transient failures worth retrying with backoff: infrastructure, not input.
TRANSIENT_ERRORS = (SQLAlchemyError, ConnectionError, TimeoutError, OSError)

celery_app = Celery("rag_engine", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    task_time_limit=1800,
    worker_max_tasks_per_child=25,
    worker_max_memory_per_child=1_500_000,  # 1.5 GB → hard recycle
    # Fail fast on a lost broker instead of silently draining an orphan queue.
    worker_cancel_long_running_tasks_on_connection_loss=True,
)

def _fresh_session() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Engine + sessionmaker scoped to *this* event loop.

    Each task runs inside its own asyncio.run() loop. A shared async engine's
    connection pool outlives that loop: a connection handed out on task N was
    created on task N-1's loop, and asyncpg binds its futures to the loop that
    created them — reusing it raises "Future attached to a different loop".
    Building the engine per task is microseconds of overhead against a pool of
    one connection and makes the loop lifecycle explicit.
    """
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@celery_app.task(
    bind=True,
    max_retries=3,
    autoretry_for=TRANSIENT_ERRORS,
    retry_backoff=True,  # 1s, 2s(±jitter), 4s …
    retry_backoff_max=300,
    retry_jitter=True,
)
def ingest_document(
    self, path: str, title: str, authority: str = "unknown", workspace: str = "default"
) -> dict:
    try:
        result = asyncio.run(_ingest_async(path, title, authority, workspace))
    except TRANSIENT_ERRORS:
        # Typed for autoretry_for: Celery re-raises, retries with backoff, and
        # the exception surfaces in /tasks only after the final attempt.
        raise
    except Exception as exc:
        logger.exception("Ingest failed permanently for %s", path)
        raise RuntimeError(f"failed to ingest {Path(path).name}: {exc}") from exc
    # The upload is named by its content hash, so any pending task pointing at
    # the same path can re-materialize nothing — it is safe to remove the file
    # once ingestion succeeded. Failed jobs keep the file for inspection (and
    # for the retry to consume).
    Path(path).unlink(missing_ok=True)
    return result


async def _ingest_async(path: str, title: str, authority: str, workspace: str) -> dict:
    engine, Session = _fresh_session()
    try:
        return await _ingest_with_session(engine, Session, path, title, authority, workspace)
    finally:
        # Closes this loop's connections; never touches a previous loop's pool.
        await engine.dispose()


async def _ingest_with_session(
    engine: AsyncEngine,
    Session: async_sessionmaker[AsyncSession],
    path: str,
    title: str,
    authority: str,
    workspace: str,
) -> dict:
    if not Path(path).exists():
        # Content-hash filenames mean this path identifies one specific file.
        # A sibling task ingesting the SAME bytes can succeed and unlink the
        # upload while this task sits between retries — if that sibling
        # committed, the work is done and the right answer is "skipped".
        async with Session() as session:
            existing = (
                await session.execute(
                    select(Document).where(Document.source_uri == path)
                )
            ).scalar_one_or_none()
        if existing:
            return {"status": "skipped", "doc_id": str(existing.id)}
        raise FileNotFoundError(path)

    file_hash = _sha256_file(path)

    async with Session() as session:
        existing = (
            await session.execute(select(Document).where(Document.source_uri == path))
        ).scalar_one_or_none()

        if existing and existing.content_hash == file_hash:
            return {"status": "skipped", "doc_id": str(existing.id)}

        if existing:
            await session.execute(delete(Chunk).where(Chunk.doc_id == existing.id))
            doc = existing
            doc.content_hash = file_hash
            doc.title = title
            doc.authority = authority
            doc.workspace = workspace
        else:
            doc = Document(
                title=title,
                source_uri=path,
                content_hash=file_hash,
                authority=authority,
                workspace=workspace,
            )
            session.add(doc)
            await session.flush()

        blocks = parse_document(path)
        chunker = RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
        chunks = chunker.chunk(blocks)
        if not chunks:
            await session.commit()
            return {"status": "empty", "doc_id": str(doc.id)}

        # One transaction per document: the delete above plus every chunk insert
        # commit together, so a failure mid-way rolls the document back to its
        # previous state instead of leaving it half-indexed.
        # Memory stays O(batch): each batch is sent to Postgres immediately via
        # a bulk insert; nothing accumulates in the session's identity map.
        # on_conflict_do_nothing rides on chunks' UNIQUE (doc_id, content_hash)
        # constraint, so a duplicate chunk *within this document* is skipped
        # instead of failing the whole batch. (Cross-document duplicates are
        # allowed now: different workspaces may store the same text.)

        # With pii_masking="all" PII is removed BEFORE hashing/embedding/storing:
        # the stored text, its content_hash, and its embedding all describe the
        # masked text, so retrieval stays consistent. Note this makes at-rest
        # masking irreversible — the original is never persisted.
        mask_at_rest = settings.pii_masking == "all"
        mask_style = settings.pii_mask_style
        if mask_at_rest:
            for c in chunks:
                c["text"] = mask_pii(c["text"], style=mask_style)

        seen: set[str] = set()
        written = 0
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors = embed_texts([c["text"] for c in batch])
            rows = []
            for c, vec in zip(batch, vectors, strict=True):
                h = _sha256(c["text"])
                if h in seen:
                    continue
                seen.add(h)
                rows.append(
                    {
                        "doc_id": doc.id,
                        "chunk_index": c["metadata"].get("chunk_index", 0),
                        "content": c["text"],
                        "content_hash": h,
                        "token_count": len(_ENC.encode(c["text"])),
                        # workspace travels on the chunk metadata so query-time
                        # scoping rides the existing metadata @> GIN filter.
                        "metadata": {**c["metadata"], "workspace": workspace},
                        "embedding": vec,
                    }
                )
            if not rows:
                continue
            insert_result = await session.execute(
                pg_insert(Chunk)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["doc_id", "content_hash"])
            )
            # Session.execute is statically typed as Result, which doesn't
            # declare rowcount; DML statements actually return a CursorResult,
            # where rowcount counts only rows PostgreSQL really inserted
            # (rows skipped by ON CONFLICT DO NOTHING are excluded).
            written += cast(CursorResult, insert_result).rowcount or 0

        await session.commit()
        return {"status": "indexed", "doc_id": str(doc.id), "chunks": written}
