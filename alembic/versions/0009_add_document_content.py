"""add content column to documents for parent-child retrieval

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-16
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS content TEXT
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS content")
