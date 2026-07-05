import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
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


def _is_retryable_structured(exc: BaseException) -> bool:
    """结构化输出重试谓词:schema 解析/校验失败重新采样通常可修复,也视为可重试。"""
    if isinstance(exc, (OutputParserException, ValidationError)):
        return True
    return is_retryable(exc)


def _schema_instruction(schema: type[BaseModel]) -> SystemMessage:
    """json_mode 不会把 schema 传给模型,须在消息中显式给出输出结构。"""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return SystemMessage(
        content=(
            "仅输出一个 JSON 对象,不要包含任何其他文字,也不要用代码块包裹。"
            f"输出必须符合以下 JSON Schema:\n{schema_json}"
        )
    )


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
            api_key=settings.MODEL_KEY,
            model=settings.MODEL_NAME,
            base_url=settings.MODEL_URL,
            temperature=0,
            seed=42,
        )
        self._model_name = settings.MODEL_NAME

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

    async def ainvoke_structured(
        self, messages: list[BaseMessage | str], schema: type[BaseModel]
    ) -> BaseModel:
        """带重试的结构化输出调用:json_mode + schema 注入。

        DeepSeek thinking 模式不支持强制 tool_choice(function_calling 400),
        json_schema 未开放(400);故改用 json_mode,并在消息中显式注入 JSON Schema,
        否则模型不知道字段名会自行发挥,导致 Pydantic 校验失败。
        """
        structured = self._model.with_structured_output(schema, method="json_mode")
        # schema 说明插在开头的 system 消息之后:部分 OpenAI 兼容端点要求 system 在前。
        msgs = list(messages)
        i = 0
        while i < len(msgs) and isinstance(msgs[i], SystemMessage):
            i += 1
        msgs.insert(i, _schema_instruction(schema))
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
            retry=retry_if_exception(_is_retryable_structured),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                async with self._translate():
                    result = await structured.ainvoke(msgs)
                    if result is None:
                        raise OutputParserException("模型未返回结构化输出(未调用工具)")
                    return result
        raise AssertionError("unreachable")

    async def astream(self, messages: list[BaseMessage | str]):
        """单次异步流式（不重试，流式中途重试语义复杂，留待 C）。"""
        async with self._translate():
            async for chunk in self._model.astream(messages):
                yield chunk

