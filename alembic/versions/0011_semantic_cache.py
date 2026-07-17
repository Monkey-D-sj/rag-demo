"""semantic_cache - 语义缓存(答案级,全局作用域)

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-17
"""
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id         BIGSERIAL PRIMARY KEY,
            question   text NOT NULL,
            answer     text NOT NULL,
            citations  jsonb NOT NULL DEFAULT '[]'::jsonb,
            embedding  vector(1024) NOT NULL,
            hit_count  int NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding"
        " ON semantic_cache USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS semantic_cache")
