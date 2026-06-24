from fastapi import Request

from rag.memory import MemoryManager
from rag.models.base import ChatModel


def get_memory_manager(request: Request) -> MemoryManager:
    return request.app.state.memory_manager


def get_llm(request: Request) -> ChatModel:
    return request.app.state.llm
