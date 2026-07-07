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

	async def add_message(
		self, session_id: str, text: str, metadata: dict | None = None
	) -> None: ...


class MessageRole(str, Enum):
	"""消息角色：每个成员自带中文标签，display 为 property。"""

	USER = ("user", "用户")
	ASSISTANT = ("assistant", "AI")

	def __new__(cls, value: str, label: str):
		obj = str.__new__(cls, value)
		obj._value_ = value
		obj._label_ = label
		return obj

	@property
	def display(self) -> str:
		"""角色的中文显示标签。"""
		return self._label_

	@classmethod
	def display_of(cls, value: str | None) -> str | None:
		"""从存储值（如 "user"）反查中文标签；未知值返回 None。"""
		if value is None:
			return None
		try:
			return cls(value).display
		except ValueError:
			return None


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
	is_out_of_scope: bool  # True 表示查询与知识库无关，跳过召回直接大模型兜底

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

	