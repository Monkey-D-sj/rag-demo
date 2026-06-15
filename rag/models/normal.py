import os
import logging
from typing import Any, Generator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)

from rag.common.exception import LLMRequestException
from rag.models.base import ChatModel

load_dotenv()

logger = logging.getLogger(__name__)

# 共享的重试配置
_retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class NormalModel(ChatModel):
    def __init__(self):
        self._model = ChatOpenAI(
            api_key=os.getenv("MODEL_KEY"),
            model=os.getenv("MODEL_NAME"),
            base_url=os.getenv("MODEL_URL"),
            temperature=0,
            seed=42,
        )

    def bind_tools(self, tools: list[BaseTool]):
        self._model = self._model.bind_tools(tools)

    @_retry_decorator
    def _do_invoke(self, messages: list[Any | str]) -> str:
        return self._model.invoke(messages).content

    def invoke(self, messages: list[Any | str]) -> str:
        """单次调用 LLM，不重试"""
        return self._model.invoke(messages).content

    def stream(self, messages: list[Any | str]) -> Generator[AIMessageChunk, Any, None]:
        """单次流式调用 LLM，不重试"""
        yield from self._model.stream(messages)

    def invoke_with_retry(self, messages: list[Any | str]) -> str:
        """调用 LLM 模型，自动重试"""
        try:
            return self._do_invoke(messages)
        except RetryError as e:
            raise LLMRequestException(
                "invoke failed after 3 retries",
                model=getattr(self._model, "model_name", ""),
                attempts=3,
            ) from e.last_attempt.exception()

    def stream_with_retry(
        self, messages: list[Any | str]
    ) -> Generator[AIMessageChunk | str, Any, None]:
        """流式调用 LLM，失败时降级到非流式兜底"""
        try:
            yield from self._model.stream(messages)
            return
        except Exception as e:
            logger.warning("stream interrupted, falling back to invoke: %s", e)

        yield AIMessageChunk(content="\n\n[响应中断，正在恢复...]\n\n")
        try:
            result = self._do_invoke(messages)
            yield AIMessageChunk(content=result)
        except RetryError as e:
            raise LLMRequestException(
                "invoke fallback failed after 3 retries",
                model=getattr(self._model, "model_name", ""),
                attempts=3,
            ) from e.last_attempt.exception()
