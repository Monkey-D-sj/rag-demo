from contextlib import asynccontextmanager
from typing import Any, Callable

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.common.exception import LLMException, from_http_error, is_retryable
from rag.common.logging import get_logger
from rag.config import Settings
from rag.models.base import ChatModel

logger = get_logger()

ErrorHandler = Callable[[LLMException], None]
ErrorHandlers = dict[int | str, ErrorHandler]


def dispatch_error(exc: LLMException, handlers: ErrorHandlers | None) -> None:
    """按 status_code 分发，支持 '*' 通配。"""
    if not handlers:
        return
    handler = handlers.get(exc.status_code) or handlers.get("*")
    if handler:
        try:
            handler(exc)
        except Exception:
            logger.exception("error handler failed")


def _extract_status_code(exc: BaseException) -> int:
    if hasattr(exc, "status_code"):
        return exc.status_code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code
    return 0


class NormalModel(ChatModel):
    def __init__(self, settings: Settings):
        self._model = ChatOpenAI(
            api_key=settings.model_key,
            model=settings.model_name,
            base_url=settings.model_url,
            temperature=0,
            seed=42,
        )
        self._model_name = settings.model_name

    def bind_tools(self, tools: list[BaseTool]) -> None:
        self._model = self._model.bind_tools(tools)

    @asynccontextmanager
    async def _translate(self):
        try:
            yield
        except LLMException:
            raise
        except Exception as e:
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise

    async def ainvoke(self, messages: list[BaseMessage | str]) -> str:
        """带重试的异步调用。"""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
            retry=retry_if_exception(is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                async with self._translate():
                    rsp = await self._model.ainvoke(messages)
                    return rsp.content
        raise AssertionError("unreachable")

    async def astream(self, messages: list[BaseMessage | str]):
        """单次异步流式（不重试，流式中途重试语义复杂，留待 C）。"""
        async with self._translate():
            async for chunk in self._model.astream(messages):
                yield chunk

