"""add retry_count to documents for DLQ self-healing

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-25
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS retry_count INT NOT NULL DEFAULT 0
        """
    )
    # 加速 cron 扫描: WHERE status='failed' AND retry_count < N ORDER BY updated_at
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_failed_retry
            ON documents (status, retry_count, updated_at)
            WHERE status = 'failed'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_documents_failed_retry")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS retry_count")
