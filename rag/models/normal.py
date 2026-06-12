import os
import time
from typing import Any, Generator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

from rag.models.base import ChatModel

load_dotenv()

class NormalModel(ChatModel):
	def __init__(self):
		self._model = ChatOpenAI(
			api_key=os.getenv("MODEL_KEY"),
			model=os.getenv("MODEL_NAME"),
			base_url=os.getenv("MODEL_URL"),
			temperature=0,
			seed=42,
		)
	
	def llm_invoke(self, messages: list[Any | str]) -> str:
		"""调用 LLM 模型"""
		for i in range(3):
			try:
				res = self.invoke(messages).content
				return res
			except Exception as e:
				time.sleep(2 ** i)
		return "llm invoke failed"
	
	def llm_stream(self, messages: list[Any | str]) -> Generator[AIMessageChunk | str, Any, None]:
		"""调用 LLM 模型，返回流式结果"""
		for i in range(3):
			try:
				for chunk in self._model.stream(messages):
					yield chunk
			except Exception as e:
				time.sleep(2 ** i)
				time.sleep(2)
		yield "llm stream failed"

