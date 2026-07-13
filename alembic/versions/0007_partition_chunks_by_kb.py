"""partition document_chunks by LIST(knowledge_base_id)

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-13
"""
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024
DEFAULT_KB_ID = "00000000-0000-0000-0000-000000000001"
EVAL_KB_ID = "00000000-0000-0000-0000-0000000000ee"
PARTITIONED_TABLE = "document_chunks"
PARENT_NEW = "document_chunks_partitioned"
BACKUP_TABLE = "document_chunks_old"


def upgrade() -> None:
    # 1. 创建分区父表
    op.execute(
        f"""
        CREATE TABLE {PARENT_NEW} (
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

    # 2. 为已知 KB 创建分区 + 默认分区
    for kb_id, suffix in [
        (DEFAULT_KB_ID, "default_kb"),
        (EVAL_KB_ID, "eval_kb"),
    ]:
        safe_suffix = suffix.replace("-", "_")
        op.execute(
            f"""
            CREATE TABLE dchunks_{safe_suffix} PARTITION OF {PARENT_NEW}
                FOR VALUES IN ('{kb_id}')
            """
        )

    op.execute(
        f"""
        CREATE TABLE dchunks_other PARTITION OF {PARENT_NEW}
            DEFAULT
        """
    )

    # 3. 迁移数据
    op.execute(
        f"""
        INSERT INTO {PARENT_NEW} (id, document_id, knowledge_base_id,
                                   chunk_index, text, embedding, metadata, created_at)
        SELECT id, document_id, knowledge_base_id,
               chunk_index, text, embedding, metadata, created_at
        FROM {PARTITIONED_TABLE}
        """
    )

    # 4. 原子替换
    op.execute(f"ALTER TABLE {PARTITIONED_TABLE} RENAME TO {BACKUP_TABLE}")
    op.execute(f"ALTER TABLE {PARENT_NEW} RENAME TO {PARTITIONED_TABLE}")
    op.execute(f"ALTER INDEX {PARENT_NEW}_pkey RENAME TO {PARTITIONED_TABLE}_pkey")

    # 5. 重建 FK
    op.execute(
        f"""
        ALTER TABLE {PARTITIONED_TABLE}
            ADD CONSTRAINT dchunks_doc_fk FOREIGN KEY (document_id)
            REFERENCES documents(id) ON DELETE CASCADE
        """
    )

    # 6. 重建索引（建在父表上，pgvector >= 0.7 自动传播到各分区）
    op.execute(
        f"""
        CREATE INDEX idx_dchunks_embedding
            ON {PARTITIONED_TABLE} USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_dchunks_doc
            ON {PARTITIONED_TABLE} (document_id)
        """
    )

    # BM25: Paradedb 分区表支持因版本而异；尝试建在父表，失败则逐分区建
    try:
        op.execute(
            f"""
            CREATE INDEX idx_dchunks_bm25
                ON {PARTITIONED_TABLE} USING bm25 (id, text, knowledge_base_id)
                WITH (
                    key_field='id',
                    text_fields='{{"text": {{"tokenizer": {{"type": "jieba"}}}}}}'
                )
            """
        )
    except Exception:
        # 降级：在每个子分区上单独建 BM25 索引
        for partition in ["dchunks_default_kb", "dchunks_eval_kb", "dchunks_other"]:
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
                pass  # BM25 不可用时静默降级，向量路仍是主路径

    # 7. 删除旧表（数据已迁移）
    op.execute(f"DROP TABLE {BACKUP_TABLE} CASCADE")


def downgrade() -> None:
    # 逆向：重建普通表 → 回迁数据 → 替换 → 重建原始索引

    op.execute(
        f"""
        CREATE TABLE {PARENT_NEW} (
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

    # 从分区表回迁数据
    op.execute(
        f"""
        INSERT INTO {PARENT_NEW} (id, document_id, knowledge_base_id,
                                   chunk_index, text, embedding, metadata, created_at)
        SELECT id, document_id, knowledge_base_id,
               chunk_index, text, embedding, metadata, created_at
        FROM {PARTITIONED_TABLE}
        """
    )

    # 替换
    op.execute(f"DROP TABLE {PARTITIONED_TABLE} CASCADE")
    op.execute(f"ALTER TABLE {PARENT_NEW} RENAME TO {PARTITIONED_TABLE}")

    # 重建索引（原始 0002 + 0005 定义）
    op.execute(
        f"""
        CREATE INDEX idx_dchunks_embedding
            ON {PARTITIONED_TABLE} USING hnsw (embedding vector_cosine_ops)
        """
    )
    try:
        op.execute(
            f"""
            CREATE INDEX idx_dchunks_bm25
                ON {PARTITIONED_TABLE} USING bm25 (id, text, knowledge_base_id)
                WITH (
                    key_field='id',
                    text_fields='{{"text": {{"tokenizer": {{"type": "jieba"}}}}}}'
                )
            """
        )
    except Exception:
        pass
    op.execute(
        f"""
        CREATE INDEX idx_dchunks_doc
            ON {PARTITIONED_TABLE} (document_id)
        """
    )
