from abc import ABC, abstractmethod


class ShortTermMemoryAdapter(ABC):
    """短期记忆适配器 — 会话级，轻量快速，自动过期"""

    @abstractmethod
    def add(self, session_id: str, text: str, metadata: dict | None = None) -> None:
        """向会话添加一条消息"""
        ...

    @abstractmethod
    def get_recent(self, session_id: str, n: int = 10) -> list[dict]:
        """获取会话最近 N 条消息"""
        ...

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """清除会话所有消息"""
        ...


class LongTermMemoryAdapter(ABC):
    """长期记忆适配器 — 持久化，向量+BM25 检索，支持增删改查"""

    @abstractmethod
    def add(self, session_id: str, text: str, metadata: dict | None = None) -> str:
        """添加一条记忆，自动生成 embedding，返回 memory_id"""
        ...

    @abstractmethod
    def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        """基于向量相似度搜索记忆，可按 metadata 字段过滤"""
        ...

    @abstractmethod
    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 关键词检索"""
        ...

    @abstractmethod
    def update(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        """更新记忆，自动重新生成 embedding"""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """删除一条记忆"""
        ...

    @abstractmethod
    def get_by_time(self, start: float, end: float) -> list[dict]:
        """按时间范围查询记忆（start/end 为 Unix timestamp）"""
        ...
