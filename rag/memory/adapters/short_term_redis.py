import json
from typing import Optional

from rag.db.redis import get_redis_client
from rag.memory.adapters.base import ShortTermMemoryAdapter


class RedisShortTermMemory(ShortTermMemoryAdapter):
    """基于 Redis List 的短期记忆

    - 按 session_id 隔离，每个 session 一个 List
    - 自动 TTL 过期（默认 24h）
    - 保留最近 N 条消息（默认 50 条）
    """

    def __init__(
        self,
        max_messages: int = 50,
        ttl_seconds: int = 24 * 60 * 60,  # 24 hours
    ):
        self._redis = get_redis_client()
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    def add(
        self, session_id: str, text: str, metadata: Optional[dict] = None
    ) -> None:
        """向会话追加一条消息，超出上限时自动裁剪最早的消息"""
        key = self._session_key(session_id)
        payload = json.dumps({
            "text": text,
            "metadata": metadata or {},
        }, ensure_ascii=False)

        pipe = self._redis.pipeline()
        pipe.rpush(key, payload)                      # 追加到末尾
        pipe.ltrim(key, -self.max_messages, -1)       # 保留最近 N 条
        pipe.expire(key, self.ttl_seconds)            # 刷新 TTL
        pipe.execute()

    def get_recent(self, session_id: str, n: int = 10) -> list[dict]:
        """获取会话最近 N 条消息（按时间升序）"""
        if n <= 0:
            return []

        key = self._session_key(session_id)
        raw = self._redis.lrange(key, -n, -1)
        return [json.loads(item) for item in raw]

    def clear(self, session_id: str) -> None:
        """清除会话全部消息"""
        self._redis.delete(self._session_key(session_id))
