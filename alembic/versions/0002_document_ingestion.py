"""document ingestion: knowledge_bases + documents + document_chunks

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-24
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024
DEFAULT_KB_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        INSERT INTO knowledge_bases (id, name)
        VALUES ('{DEFAULT_KB_ID}', 'default')
        ON CONFLICT (id) DO NOTHING
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id),
            filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            content_hash TEXT,
            object_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            chunk_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            knowledge_base_id UUID NOT NULL,
            chunk_index INT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_bm25
            ON document_chunks USING bm25 (id, text, metadata)
            WITH (key_field='id')
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dchunks_doc ON document_chunks (document_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_dchunks_bm25")
    op.execute("DROP INDEX IF EXISTS idx_dchunks_embedding")
    op.execute("DROP TABLE IF EXISTS document_chunks")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS knowledge_bases")
