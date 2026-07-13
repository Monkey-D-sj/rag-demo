"""remove default KB, seed novel + regulation KBs

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-13
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024
NOVEL_KB_ID = "00000000-0000-0000-0000-000000000002"
REGULATION_KB_ID = "00000000-0000-0000-0000-000000000003"


def upgrade() -> None:
    # 0. 确保扩展存在（兼容从 stamp 0007 开始运行的场景）
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")

    # 1. 清空旧数据
    op.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
    op.execute("DROP TABLE IF EXISTS knowledge_bases CASCADE")

    # 2. 重建 knowledge_bases
    op.execute(
        """
        CREATE TABLE knowledge_bases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )

    # 3. 种子两条 KB
    op.execute(
        f"""
        INSERT INTO knowledge_bases (id, name) VALUES
            ('{NOVEL_KB_ID}', '小说'),
            ('{REGULATION_KB_ID}', '法规')
        """
    )

    # 4. 重建 documents
    op.execute(
        """
        CREATE TABLE documents (
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
            graph_status TEXT NOT NULL DEFAULT 'pending',
            graph_error TEXT,
            retry_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )

    # 5. 重建分区父表
    op.execute(
        f"""
        CREATE TABLE document_chunks (
            id UUID DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL,
            knowledge_base_id UUID NOT NULL,
            chunk_index INT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (id, knowledge_base_id)
        ) PARTITION BY LIST (knowledge_base_id)
        """
    )

    # 6. 为两个 KB 创建分区（无 DEFAULT 分区 — 新 KB 由 ensure_kb_partition() runtime 建）
    op.execute(
        f"""
        CREATE TABLE dchunks_novel PARTITION OF document_chunks
            FOR VALUES IN ('{NOVEL_KB_ID}')
        """
    )
    op.execute(
        f"""
        CREATE TABLE dchunks_regulation PARTITION OF document_chunks
            FOR VALUES IN ('{REGULATION_KB_ID}')
        """
    )

    # 7. FK + 索引
    op.execute(
        """
        ALTER TABLE document_chunks
            ADD CONSTRAINT dchunks_doc_fk FOREIGN KEY (document_id)
            REFERENCES documents(id) ON DELETE CASCADE
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_dchunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        "CREATE INDEX idx_dchunks_doc ON document_chunks (document_id)"
    )

    # BM25: 尝试建在父表，失败则逐分区建
    try:
        op.execute(
            f"""
            CREATE INDEX idx_dchunks_bm25
                ON document_chunks USING bm25 (id, text, knowledge_base_id)
                WITH (
                    key_field='id',
                    text_fields='{{"text": {{"tokenizer": {{"type": "jieba"}}}}}}'
                )
            """
        )
    except Exception:
        for partition in ["dchunks_novel", "dchunks_regulation"]:
            try:
                op.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_dchunks_bm25_{partition}
                        ON {partition} USING bm25 (id, text, knowledge_base_id)
                        WITH (
                            key_field='id',
                            text_fields='{{"text": {{"tokenizer": {{"type": "jieba"}}}}}}'
                        )
                    """
                )
            except Exception:
                pass


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
    op.execute("DROP TABLE IF EXISTS knowledge_bases CASCADE")
