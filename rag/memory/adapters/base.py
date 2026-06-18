from abc import ABC, abstractmethod


class ShortTermMemoryAdapter(ABC):
    """短期记忆适配器 — 会话级，轻量快速，自动过期。"""

    @abstractmethod
    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> None: ...

    @abstractmethod
    async def get_recent(self, session_id: str, n: int = 10) -> list[dict]: ...

    @abstractmethod
    async def clear(self, session_id: str) -> None: ...


class LongTermMemoryAdapter(ABC):
    """长期记忆适配器 — 持久化，向量 + BM25 检索。"""

    @abstractmethod
    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> str: ...

    @abstractmethod
    async def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]: ...

    @abstractmethod
    async def update(self, memory_id: str, text: str, metadata: dict | None = None) -> None: ...

    @abstractmethod
    async def delete(self, memory_id: str) -> None: ...
