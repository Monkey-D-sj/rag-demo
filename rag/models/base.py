from abc import ABC
from typing import Any


class ChatModel(ABC):
	def invoke(self, messages: list[Any | str]):
		...
	
	def stream(self, messages: list[Any | str]):
		...