"""long_term_memories: session_id 提为独立列 + B-tree 索引

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-06
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE long_term_memories
        ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT ''
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ltm_session_id
        ON long_term_memories (session_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_ltm_session_id")
    op.execute(
        "ALTER TABLE long_term_memories DROP COLUMN IF EXISTS session_id"
    )
