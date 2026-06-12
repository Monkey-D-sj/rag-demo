import uuid
from typing import Optional

from rag.db.postgres import ensure_pgbm25_extension, ensure_pgvector_extension, get_cursor
from rag.models.embedding import generate_embeddings_batch
from rag.memory.adapters.base import LongTermMemoryAdapter
from psycopg2.extras import Json

# pgvector 默认的 embedding 维度
EMBEDDING_DIM = 1024

# 建表 SQL（幂等）
_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS long_term_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}),
    metadata JSONB DEFAULT '{{}}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ltm_embedding
    ON long_term_memories
    USING hnsw (embedding vector_cosine_ops)
"""

# pg_bm25 倒排索引（ParadeDB，幂等：重复创建会抛异常，外层 catch）
_CREATE_BM25_INDEX_SQL = """
CALL paradedb.create_bm25(
    table_name => 'long_term_memories',
    index_name => 'idx_ltm_bm25',
    key_field  => 'id',
    text_fields => '{text}',
    json_fields => '{metadata}'
)
"""


class PgVectorLongTermMemory(LongTermMemoryAdapter):
    """基于 PostgreSQL + pgvector + pg_bm25 的长期记忆

    - 自动建表、建索引（幂等）
    - 向量相似度检索（余弦距离）
    - BM25 关键词检索
    - 支持增删改查 + 时间范围查询
    """

    def __init__(self) -> None:
        ensure_pgvector_extension()
        ensure_pgbm25_extension()
        with get_cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
            cur.execute(_CREATE_INDEX_SQL)
            try:
                cur.execute(_CREATE_BM25_INDEX_SQL)
            except Exception:
                pass  # 索引已存在则忽略

    # ── 公开接口 ──────────────────────────────────────

    def add(
        self, session_id: str, text: str, metadata: Optional[dict] = None
    ) -> str:
        """添加一条记忆，自动生成 embedding，返回 memory_id"""
        memory_id = str(uuid.uuid4())
        embedding = generate_embeddings_batch([text])[0]
        merged_metadata = {"session_id": session_id, **(metadata or {})}

        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO long_term_memories (id, text, embedding, metadata)
                VALUES (%(id)s, %(text)s, %(embedding)s, %(metadata)s)
                """,
                {
                    "id": memory_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": Json(merged_metadata),
                },
            )
        return memory_id

    def search(
        self, query: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> list[dict]:
        """基于向量相似度搜索记忆，可按 metadata 字段过滤

        Args:
            query: 查询文本
            top_k: 返回条数
            filters: 可选的 metadata 过滤条件，如 {"category": "法规", "year": 2025}
        """
        query_embedding = generate_embeddings_batch([query])[0]

        sql = """
            SELECT id, text, metadata, created_at,
                   1 - (embedding <=> %(embedding)s) AS similarity
            FROM long_term_memories
            WHERE 1=1
        """
        params: dict = {"embedding": query_embedding, "top_k": top_k}

        # 动态拼接 metadata 过滤
        if filters:
            for i, (key, value) in enumerate(filters.items()):
                param_name = f"filter_{i}"
                sql += f" AND metadata->>%(key_{i})s = %({param_name})s"
                params[f"key_{i}"] = key
                params[param_name] = str(value)

        sql += " ORDER BY embedding <=> %(embedding)s LIMIT %(top_k)s"

        with get_cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 关键词检索

        Args:
            query: 搜索查询文本
            top_k: 返回条数
        """
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, text, metadata, created_at,
                       paradedb.score(id) AS bm25_score
                FROM long_term_memories
                WHERE long_term_memories @@@ paradedb.parse(%(query)s)
                ORDER BY bm25_score DESC
                LIMIT %(top_k)s
                """,
                {"query": query, "top_k": top_k},
            )
            return cur.fetchall()

    def update(
        self, memory_id: str, text: str, metadata: Optional[dict] = None
    ) -> None:
        """更新记忆文本，自动重新生成 embedding"""
        embedding = generate_embeddings_batch([text])[0]
        metadata_payload = Json(metadata) if metadata is not None else None

        with get_cursor() as cur:
            cur.execute(
                """
                UPDATE long_term_memories
                SET text = %(text)s,
                    embedding = %(embedding)s,
                    metadata = CASE
                        WHEN %(metadata)s IS NULL THEN long_term_memories.metadata
                        ELSE COALESCE(long_term_memories.metadata, '{}'::jsonb) || %(metadata)s::jsonb
                    END,
                    updated_at = now()
                WHERE id = %(id)s
                """,
                {
                    "id": memory_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata_payload,
                },
            )

    def delete(self, memory_id: str) -> None:
        """删除一条记忆"""
        with get_cursor() as cur:
            cur.execute(
                "DELETE FROM long_term_memories WHERE id = %(id)s",
                {"id": memory_id},
            )

    def get_by_time(self, start: float, end: float) -> list[dict]:
        """按时间范围查询记忆

        Args:
            start: 起始 Unix timestamp
            end: 结束 Unix timestamp
        """
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, text, metadata, created_at, updated_at
                FROM long_term_memories
                WHERE created_at >= to_timestamp(%(start)s)
                  AND created_at <= to_timestamp(%(end)s)
                ORDER BY created_at DESC
                """,
                {"start": start, "end": end},
            )
            return cur.fetchall()
