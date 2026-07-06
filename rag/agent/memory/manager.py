import json
import uuid

from pgvector import Vector
from psycopg.types.json import Json

from rag.db.postgres import get_cursor
from rag.models.embedding import EmbeddingModel


class MemoryManager:
    """统一管理短期/长期记忆，直接操作 Redis + PostgreSQL(pgvector)。"""

    def __init__(
        self,
        pool,
        embedding: EmbeddingModel,
        redis=None,
        *,
        short_max_messages: int = 10,
        short_ttl_seconds: int = 24 * 60 * 60,
    ):
        self._pool = pool
        self._embedding = embedding
        self._redis = redis
        self._short_max_messages = short_max_messages
        self._short_ttl_seconds = short_ttl_seconds

    # ── 短期记忆 (Redis) ──

    @staticmethod
    def _short_key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def add_message(
        self, session_id: str, text: str, metadata: dict | None = None
    ) -> None:
        if self._redis is None:
            return
        key = self._short_key(session_id)
        payload = json.dumps(
            {"text": text, "metadata": metadata or {}}, ensure_ascii=False
        )
        pipe = self._redis.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self._short_max_messages, -1)
        pipe.expire(key, self._short_ttl_seconds)
        await pipe.execute()

    async def get_recent_messages(self, session_id: str, n: int = 10) -> list[dict]:
        if self._redis is None or n <= 0:
            return []
        raw = await self._redis.lrange(self._short_key(session_id), -n, -1)
        return [json.loads(item) for item in raw]

    async def clear_session(self, session_id: str) -> None:
        if self._redis is not None:
            await self._redis.delete(self._short_key(session_id))

    # ── 长期记忆 (PostgreSQL + pgvector) ──

    async def add(
        self, session_id: str, text: str, metadata: dict | None = None
    ) -> str:
        memory_id = str(uuid.uuid4())
        embedding = (await self._embedding.embed([text]))[0]
        async with get_cursor(self._pool) as cur:
            await cur.execute(
                """
                INSERT INTO long_term_memories (id, session_id, text, embedding, metadata)
                VALUES (%(id)s, %(session_id)s, %(text)s, %(embedding)s, %(metadata)s)
                """,
                {
                    "id": memory_id,
                    "session_id": session_id,
                    "text": text,
                    "embedding": Vector(embedding),
                    "metadata": Json(metadata or {}),
                },
            )
        return memory_id

    async def search(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        query_embedding = (await self._embedding.embed([query]))[0]
        sql = """
            SELECT id, text, metadata, created_at,
                   1 - (embedding <=> %(embedding)s) AS similarity
            FROM long_term_memories
            WHERE session_id = %(session_id)s
        """
        params: dict = {
            "embedding": Vector(query_embedding),
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

