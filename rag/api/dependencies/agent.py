from fastapi import Request

from rag.document.retriever import KnowledgeRetriever
from rag.agent.memory import MemoryManager
from rag.models.base import ChatModel
from rag.models.rerank import QwenReranker


def get_memory_manager(request: Request) -> MemoryManager:
    return request.app.state.memory_manager


def get_llm(request: Request) -> ChatModel:
    return request.app.state.llm


def get_retriever(request: Request) -> KnowledgeRetriever:
    return request.app.state.retriever


def get_reranker(request: Request) -> QwenReranker | None:
    return getattr(request.app.state, "reranker", None)


def get_semantic_cache(request: Request):
    return getattr(request.app.state, "semantic_cache", None)
