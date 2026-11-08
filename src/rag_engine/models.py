from datetime import UTC, datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_engine.config import get_settings
from rag_engine.db import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    title: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str] = mapped_column(String(2048), unique=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    authority: Mapped[str] = mapped_column(String(32), default="unknown")
    # Tenancy boundary: every document belongs to exactly one workspace, and
    # queries are always scoped to the caller's workspace (see api/auth.py).
    # "default" covers documents ingested before workspaces existed.
    workspace: Mapped[str] = mapped_column(String(64), default="default", index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    doc_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # Unique per document (not globally): two workspaces may legitimately both
    # store the same chunk text, and each must be retrievable inside its own
    # scope. ON CONFLICT (doc_id, content_hash) rides on this constraint.
    content_hash: Mapped[str] = mapped_column(String(64))
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    # Why HALFVEC? Half-precision vectors cut memory roughly in half versus
    # vector(dim) and are supported natively by pgvector 0.7+. HNSW indexes
    # on halfvec use halfvec_cosine_ops.
    # The dimension comes from EMBED_DIM so the column can never silently
    # disagree with the configured model — changing EMBED_MODEL without
    # matching EMBED_DIM breaks at migration time, not at first query.
    embedding: Mapped[list[float] | None] = mapped_column(
        HALFVEC(get_settings().embed_dim)
    )
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
        Index("ix_chunks_tsv_gin", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
        # Workspace-scoped dedup: same text may exist once per document.
        UniqueConstraint("doc_id", "content_hash", name="uq_chunks_doc_content_hash"),
    )
