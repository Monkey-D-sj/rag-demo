from rag.memory.adapters import PgVectorLongTermMemory, RedisShortTermMemory
from rag.memory.adapters.base import LongTermMemoryAdapter, ShortTermMemoryAdapter

class MemoryManager:
    """记忆管理器 — 组合短期/长期记忆适配器，统一管理记忆的存取"""

    def __init__(
        self,
        long_term: LongTermMemoryAdapter,
        short_term: ShortTermMemoryAdapter | None = None,
    ):
        self._long_term = long_term
        self._short_term = short_term

    def add(
        self, session_id: str, text: str, metadata: dict | None = None
    ) -> str:
        """添加一条长期记忆，返回 memory_id"""
        return self._long_term.add(session_id, text, metadata)

    def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        """向量相似度搜索长期记忆"""
        return self._long_term.search(query, session_id, top_k, filters)

    # ── 短期记忆（会话级，自动过期） ──────────────────

    def get_recent_messages(self, session_id: str, n: int = 10) -> list[dict]:
        """获取会话最近 N 条消息"""
        if self._short_term:
            return self._short_term.get_recent(session_id, n)
        return []

    def clear_session(self, session_id: str) -> None:
        """清除会话所有消息"""
        if self._short_term:
            self._short_term.clear(session_id)

_memory_manager: MemoryManager | None = None

def get_memory_manager(
    long_term: LongTermMemoryAdapter | None = None,
    short_term: ShortTermMemoryAdapter | None = None,
) -> MemoryManager:
    global _memory_manager

    if _memory_manager is None:
        if long_term is None:
            long_term = PgVectorLongTermMemory()
        if short_term is None:
            short_term = RedisShortTermMemory()
        _memory_manager = MemoryManager(long_term=long_term, short_term=short_term)

    return _memory_manager
