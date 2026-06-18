from abc import ABC, abstractmethod
from typing import Any


class ChatModel(ABC):
    @abstractmethod
    async def ainvoke(self, messages: list[Any]) -> str: ...

    @abstractmethod
    def astream(self, messages: list[Any]):
        """async generator of chunks"""
        ...