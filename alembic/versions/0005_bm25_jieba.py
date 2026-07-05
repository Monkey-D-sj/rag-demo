"""rebuild document_chunks bm25 index: jieba tokenizer + kb filter field

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-05
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_dchunks_bm25")
    # 默认分词器对中文切不出词,BM25 统计无意义;jieba 词典分词已在
    # pg_search 0.24.1 容器内实测.knowledge_base_id 进索引供查询按库过滤.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_bm25
            ON document_chunks USING bm25 (id, text, knowledge_base_id)
            WITH (
                key_field='id',
                text_fields='{"text": {"tokenizer": {"type": "jieba"}}}'
            )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_dchunks_bm25")
    # 恢复 0002 的原始定义
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_bm25
            ON document_chunks USING bm25 (id, text, metadata)
            WITH (key_field='id')
        """
    )
