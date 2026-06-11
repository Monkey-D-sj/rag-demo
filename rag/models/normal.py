import os
import time
from typing import Any, Generator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

load_dotenv()

_normal_model = ChatOpenAI(
	api_key=os.getenv("MODEL_KEY"),
	model=os.getenv("MODEL_NAME"),
	base_url=os.getenv("MODEL_URL"),
	temperature=0,
	seed=42,
)

def llm_invoke(prompt: str) -> str:
	"""调用 LLM 模型"""
	for i in range(3):
		try:
			res = _normal_model.invoke(prompt).content
			return res
		except Exception as e:
			time.sleep(2 ** i)
	return "llm invoke failed"

def llm_stream(prompt: str) -> Generator[AIMessageChunk | str, Any, None]:
	"""调用 LLM 模型，返回流式结果"""
	for i in range(3):
		try:
			for chunk in _normal_model.stream(prompt):
				yield chunk
		except Exception as e:
			time.sleep(2 ** i)
			time.sleep(2)
	yield "llm stream failed"
