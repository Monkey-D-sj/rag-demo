"""semantic_cache - 按 session 隔离语义缓存

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-29
"""
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 旧缓存没有归属信息，不能安全迁移到任意 session；缓存可重建，因此全部清空。
    op.execute("DELETE FROM semantic_cache")
    op.execute("DROP INDEX IF EXISTS idx_semantic_cache_embedding")
    op.execute(
        "ALTER TABLE semantic_cache ADD COLUMN session_id UUID NOT NULL"
    )
    op.execute(
        "ALTER TABLE semantic_cache"
        " ADD CONSTRAINT fk_semantic_cache_session"
        " FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE"
    )
    op.execute(
        "CREATE INDEX idx_semantic_cache_session_created_at"
        " ON semantic_cache (session_id, created_at DESC)"
    )


def downgrade() -> None:
    # 去掉隔离字段前先清空，避免不同 session 的记录重新混成全局缓存。
    op.execute("DELETE FROM semantic_cache")
    op.execute("DROP INDEX IF EXISTS idx_semantic_cache_session_created_at")
    op.execute(
        "ALTER TABLE semantic_cache"
        " DROP CONSTRAINT IF EXISTS fk_semantic_cache_session"
    )
    op.execute("ALTER TABLE semantic_cache DROP COLUMN IF EXISTS session_id")
    op.execute(
        "CREATE INDEX idx_semantic_cache_embedding"
        " ON semantic_cache USING hnsw (embedding vector_cosine_ops)"
    )
