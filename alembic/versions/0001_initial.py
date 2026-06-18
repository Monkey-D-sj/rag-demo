"""initial schema: long_term_memories + extensions + indexes

Revision ID: 0001
Revises:
Create Date: 2026-06-18
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # pg_bm25 was renamed to pg_search in paradedb >= 0.15; use pg_search.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ltm_embedding
            ON long_term_memories
            USING hnsw (embedding vector_cosine_ops)
        """
    )
    # BM25 索引：paradedb >= 0.15 / pg_search 0.23+ 使用 CREATE INDEX USING bm25 语法，
    # key_field='id', 索引 text 和 metadata 列。
    # 等价于旧版 paradedb.create_bm25(table_name=>'long_term_memories',
    #   index_name=>'idx_ltm_bm25', key_field=>'id',
    #   text_fields=>'{text}', json_fields=>'{metadata}')
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ltm_bm25
            ON long_term_memories
            USING bm25 (id, text, metadata)
            WITH (key_field='id')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_ltm_bm25")
    op.execute("DROP TABLE IF EXISTS long_term_memories")
