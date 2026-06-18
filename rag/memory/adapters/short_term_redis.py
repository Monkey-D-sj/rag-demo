import json

from rag.memory.adapters.base import ShortTermMemoryAdapter


class RedisShortTermMemory(ShortTermMemoryAdapter):
    """基于 Redis List 的短期记忆，按 session 隔离，自动 TTL。"""

    def __init__(self, client, max_messages: int = 50, ttl_seconds: int = 24 * 60 * 60):
        self._redis = client
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> None:
        key = self._key(session_id)
        payload = json.dumps({"text": text, "metadata": metadata or {}}, ensure_ascii=False)
        pipe = self._redis.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, self.ttl_seconds)
        await pipe.execute()

    async def get_recent(self, session_id: str, n: int = 10) -> list[dict]:
        if n <= 0:
            return []
        raw = await self._redis.lrange(self._key(session_id), -n, -1)
        return [json.loads(item) for item in raw]

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
