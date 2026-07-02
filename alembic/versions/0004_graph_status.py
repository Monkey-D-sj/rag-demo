"""add graph_status to documents for entity extraction

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-02
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS graph_status TEXT NOT NULL DEFAULT 'pending'
        """
    )
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS graph_error TEXT
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_graph_status
            ON documents (graph_status, updated_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_documents_graph_status")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS graph_error")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS graph_status")
