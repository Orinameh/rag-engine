"""workspace tenancy + per-document chunk dedup

Revision ID: c3f8a1b9d7e2
Revises: 9ba25cc82445
Create Date: 2026-11-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c3f8a1b9d7e2"
down_revision: Union[str, Sequence[str], None] = "9ba25cc82445"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Tenancy: every document belongs to a workspace. Existing documents
    # land in the "default" workspace so nothing becomes unreachable.
    op.add_column(
        "documents",
        sa.Column(
            "workspace",
            sa.String(length=64),
            nullable=False,
            server_default="default",
        ),
    )
    op.create_index(
        op.f("ix_documents_workspace"), "documents", ["workspace"], unique=False
    )

    # 2. Chunk dedup is unique per document, not per database: two workspaces
    # may store the same chunk text, and each scope must be self-contained.
    # The old global constraint is named chunks_content_hash_key (PG default).
    op.drop_constraint("chunks_content_hash_key", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunks_doc_content_hash", "chunks", ["doc_id", "content_hash"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "uq_chunks_doc_content_hash", "chunks", type_="unique"
    )
    op.create_unique_constraint(
        # same name the original Alembic revision created
        "chunks_content_hash_key",
        "chunks",
        ["content_hash"],
    )
    op.drop_index(op.f("ix_documents_workspace"), table_name="documents")
    op.drop_column("documents", "workspace")
