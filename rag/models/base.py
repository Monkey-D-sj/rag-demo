from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ChatModel(ABC):
    @abstractmethod
    async def ainvoke(self, messages: list[Any]) -> str: ...

    @abstractmethod
    async def ainvoke_with_tools(self, messages: list[Any], tools: list[Any]) -> Any:
        """原生 function calling:返回含 tool_calls 的完整消息(实现方返回 BaseMessage)。"""

    @abstractmethod
    async def ainvoke_structured(self, messages: list[Any], schema: type[T]) -> T:
        """结构化输出调用:返回 schema 的实例,由实现方保证 schema 校验。"""
        ...

    @abstractmethod
    def astream(self, messages: list[Any]):
        """async generator of chunks"""
        ...