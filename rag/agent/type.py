from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from rag.document.retriever import KnowledgeRetriever
from rag.memory import MemoryManager
from rag.models.base import ChatModel


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
	memory_manager: MemoryManager
	retriever: KnowledgeRetriever | None = None
	
	