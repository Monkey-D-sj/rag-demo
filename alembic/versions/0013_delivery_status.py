"""delivery_status - 文档任务投递状态（broker 确认标记）

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-28
"""
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 存量行默认 'sent'：已投递过的文档不会被 relay 误重投。
    # 新建文档由 create_document 显式写 'pending'，投递确认后置 'sent'。
    op.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS delivery_status"
        " text NOT NULL DEFAULT 'sent'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS delivery_status")
