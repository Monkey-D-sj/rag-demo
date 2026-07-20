"""sessions - 会话元数据管理

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-20
"""
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title      VARCHAR(255) NOT NULL DEFAULT '新会话',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at"
        " ON sessions (updated_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS sessions")
