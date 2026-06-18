from rag.memory.adapters.base import LongTermMemoryAdapter, ShortTermMemoryAdapter


class MemoryManager:
    """组合短期/长期记忆适配器，统一管理记忆存取。"""

    def __init__(
        self,
        long_term: LongTermMemoryAdapter,
        short_term: ShortTermMemoryAdapter | None = None,
    ):
        self._long_term = long_term
        self._short_term = short_term

    # ── 长期记忆 ──
    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> str:
        return await self._long_term.add(session_id, text, metadata)

    async def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        return await self._long_term.search(session_id, query, top_k, filters)

    # ── 短期记忆 ──
    async def add_message(
        self, session_id: str, text: str, metadata: dict | None = None
    ) -> None:
        if self._short_term:
            await self._short_term.add(session_id, text, metadata)

    async def get_recent_messages(self, session_id: str, n: int = 10) -> list[dict]:
        if self._short_term:
            return await self._short_term.get_recent(session_id, n)
        return []

    async def clear_session(self, session_id: str) -> None:
        if self._short_term:
            await self._short_term.clear(session_id)
