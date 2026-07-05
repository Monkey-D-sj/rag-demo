from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypedDict, runtime_checkable

from rag.models.base import ChatModel


@runtime_checkable
class RetrieverProtocol(Protocol):
	"""agent 层所需的检索器接口。具体实现(如 KnowledgeRetriever)只需满足此协议即可。"""

	async def search(
		self, query: str, knowledge_base_id: str, top_k: int = 5
	) -> list[dict]: ...


@runtime_checkable
class MemoryManagerProtocol(Protocol):
	"""agent 层所需的记忆管理器接口。具体实现(如 MemoryManager)只需满足此协议即可。"""

	async def search(
		self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
	) -> list[dict]: ...

	async def get_recent_messages(self, session_id: str, n: int = 10) -> list[dict]: ...


class StreamEventType(str, Enum):
	STATUS = "status"
	MESSAGE = "message"
	ERROR = "error"


def stream_event(type: StreamEventType, data: str) -> dict:
	return {"type": type.value, "data": data}


class MyState(TypedDict):
	session_id: str

	# ----------- 检索 -----------
	raw_query: str
	context: str
	rewrite_query: str

	# ----------- 召回 -----------
	recall_bm25_results: list[dict]
	recall_vec_results: list[dict]

	# ----------- 生成 -----------
	generated: str

@dataclass
class ContextSchema:
	llm: ChatModel
	memory_manager: MemoryManagerProtocol
	retriever: RetrieverProtocol | None = None

	