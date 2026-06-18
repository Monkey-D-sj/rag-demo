import uuid

from psycopg.types.json import Json

from rag.db.postgres import get_cursor
from rag.memory.adapters.base import LongTermMemoryAdapter
from rag.models.embedding import EmbeddingModel


class PgVectorLongTermMemory(LongTermMemoryAdapter):
    """基于 PostgreSQL + pgvector 的长期记忆（建表/索引由 alembic 负责）。"""

    def __init__(self, pool, embedding: EmbeddingModel) -> None:
        self._pool = pool
        self._embedding = embedding

    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> str:
        memory_id = str(uuid.uuid4())
        embedding = (await self._embedding.embed([text]))[0]
        merged = {"session_id": session_id, **(metadata or {})}
        async with get_cursor(self._pool) as cur:
            await cur.execute(
                """
                INSERT INTO long_term_memories (id, text, embedding, metadata)
                VALUES (%(id)s, %(text)s, %(embedding)s, %(metadata)s)
                """,
                {"id": memory_id, "text": text, "embedding": embedding, "metadata": Json(merged)},
            )
        return memory_id

    async def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        query_embedding = (await self._embedding.embed([query]))[0]
        sql = """
            SELECT id, text, metadata, created_at,
                   1 - (embedding <=> %(embedding)s) AS similarity
            FROM long_term_memories
            WHERE metadata->>'session_id' = %(session_id)s
        """
        params: dict = {
            "embedding": query_embedding,
            "session_id": session_id,
            "top_k": top_k,
        }
        if filters:
            for i, (key, value) in enumerate(filters.items()):
                sql += f" AND metadata->>%(key_{i})s = %(filter_{i})s"
                params[f"key_{i}"] = key
                params[f"filter_{i}"] = str(value)
        sql += " ORDER BY embedding <=> %(embedding)s LIMIT %(top_k)s"
        async with get_cursor(self._pool) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()

    async def update(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        embedding = (await self._embedding.embed([text]))[0]
        payload = Json(metadata) if metadata is not None else None
        async with get_cursor(self._pool) as cur:
            await cur.execute(
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
                {"id": memory_id, "text": text, "embedding": embedding, "metadata": payload},
            )

    async def delete(self, memory_id: str) -> None:
        async with get_cursor(self._pool) as cur:
            await cur.execute(
                "DELETE FROM long_term_memories WHERE id = %(id)s", {"id": memory_id}
            )
