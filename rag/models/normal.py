import asyncio
import json
import logging
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Callable, TypeVar

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

from rag.common.exception import (
    LLMException,
    LLMTimeoutError,
    from_http_error,
    is_retryable,
)
from rag.governance.config import get_governance_settings
from rag.governance.guard import LLMGuard
from rag.common.logging import get_logger
from rag.config import Settings
from rag.models.base import ChatModel

logger = get_logger()

T = TypeVar("T", bound=BaseModel)

ErrorHandler = Callable[[LLMException], None]
ErrorHandlers = dict[int | str, ErrorHandler]


def _is_retryable_structured(exc: BaseException) -> bool:
    """结构化输出重试谓词:schema 解析/校验失败重新采样通常可修复,也视为可重试。"""
    if isinstance(exc, (OutputParserException, ValidationError)):
        return True
    return is_retryable(exc)


def _schema_instruction(schema: type[BaseModel]) -> str:
    """json_mode 不会把 schema 传给模型,须在消息中显式给出输出结构。"""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return (
        "仅输出一个 JSON 对象,不要包含任何其他文字,也不要用代码块包裹。"
        f"输出必须符合以下 JSON Schema:\n{schema_json}"
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


def _usage_from(msg) -> tuple[int | None, int | None]:
    """从 LangChain 消息取 token 用量;无 usage_metadata 时返回 (None, None)。"""
    usage = getattr(msg, "usage_metadata", None) or {}
    return usage.get("input_tokens"), usage.get("output_tokens")


class NormalModel(ChatModel):
    def __init__(self, settings: Settings, guard: LLMGuard | None = None):
        gov = get_governance_settings()
        self._model = ChatOpenAI(
            api_key=settings.MODEL_KEY,
            model=settings.MODEL_NAME,
            extra_body={
                "thinking": {"type": "disabled"}
            },
            base_url=settings.MODEL_URL,
            temperature=0,
            seed=42,
            timeout=gov.LLM_TIMEOUT_SECONDS,  # 显式超时:流式为逐 chunk 读超时
            stream_usage=True,  # 流式末 chunk 携带 usage,供成本统计
            max_retries=0,  # SDK 内部重试关闭:重试统一由外层 tenacity+guard 管理,保证限流/熔断按真实请求计数
        )
        self._model_name = settings.MODEL_NAME
        self._guard = guard
        self._timeout = gov.LLM_TIMEOUT_SECONDS

    def bind_tools(self, tools: list[BaseTool]) -> None:
        self._model = self._model.bind_tools(tools)

    def _acquire(self, quota: str):
        """guard 未注入时零开销直通(null-object 模式)。"""
        if self._guard is None:
            return nullcontext()
        return self._guard.acquire(quota)

    def _track(self, call_type: str):
        """逻辑调用级统计;guard 未注入时 yield None。"""
        if self._guard is None:
            return nullcontext()
        return self._guard.track(call_type, self._model_name)

    @asynccontextmanager
    async def _translate(self):
        try:
            yield
        except LLMException:
            raise
        except TimeoutError as e:
            # asyncio.timeout 兜底触发;可重试、计入熔断
            raise LLMTimeoutError("LLM 调用超时", model=self._model_name) from e
        except Exception as e:
            # SDK 超时也翻译为 LLMTimeoutError(按类名判断,保持 is_retryable 按名判断风格)
            if type(e).__qualname__ == "APITimeoutError":
                raise LLMTimeoutError("LLM 调用超时", model=self._model_name) from e
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise

    async def ainvoke(self, messages: list[BaseMessage | str]) -> str:
        """带重试的异步调用。track 逻辑调用级,acquire 每次尝试级。"""
        async with self._track("chat") as tracker:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
                retry=retry_if_exception(is_retryable),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    if tracker is not None:
                        tracker.attempts += 1
                    async with self._acquire("chat"):
                        async with self._translate():
                            async with asyncio.timeout(self._timeout):
                                rsp = await self._model.ainvoke(messages)
                            if tracker is not None:
                                tracker.set_tokens(*_usage_from(rsp))
                            return rsp.content
            raise AssertionError("unreachable")

    async def ainvoke_structured(
        self, messages: list[BaseMessage | str], schema: type[T]
    ) -> T:
        """带重试的结构化输出调用:json_mode + schema 注入。

        DeepSeek thinking 模式不支持强制 tool_choice(function_calling 400),
        json_schema 未开放(400);故改用 json_mode,并在消息中显式注入 JSON Schema,
        否则模型不知道字段名会自行发挥,导致 Pydantic 校验失败。
        注意:temperature=0 且 seed 固定时,校验失败重试的收益依赖提供方的非确定性。
        """
        structured = self._model.with_structured_output(schema, method="json_mode")
        instruction = _schema_instruction(schema)
        # 合并进开头的 system 消息(无则插到最前):部分 OpenAI 兼容端点要求
        # system 消息唯一且在最前。str 类型消息有意按非 system 处理。
        msgs = list(messages)
        if msgs and isinstance(msgs[0], SystemMessage):
            msgs[0] = SystemMessage(content=f"{msgs[0].content}\n\n{instruction}")
        else:
            msgs.insert(0, SystemMessage(content=instruction))
        async with self._track("chat_structured") as tracker:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
                retry=retry_if_exception(_is_retryable_structured),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    if tracker is not None:
                        tracker.attempts += 1
                    async with self._acquire("chat"):
                        async with self._translate():
                            async with asyncio.timeout(self._timeout):
                                result = await structured.ainvoke(msgs)
                            if result is None:
                                raise OutputParserException("模型未返回结构化输出")
                            return result
            raise AssertionError("unreachable")

    async def astream(self, messages: list[BaseMessage | str]):
        """单次异步流式(不重试,流式中途重试语义复杂,维持现状)。"""
        async with self._track("chat_stream") as tracker:
            if tracker is not None:
                tracker.attempts = 1
            async with self._acquire("chat"):
                async with self._translate():
                    async for chunk in self._model.astream(messages):
                        usage = getattr(chunk, "usage_metadata", None)
                        if tracker is not None and usage:
                            tracker.set_tokens(
                                usage.get("input_tokens"), usage.get("output_tokens")
                            )
                        yield chunk

