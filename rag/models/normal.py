import os
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
    before_sleep_log,
    RetryError,
)

from rag.common.exception import (
    LLMException,
    from_http_error,
    is_retryable,
)
from rag.models.base import ChatModel

load_dotenv()

logger = logging.getLogger(__name__)

# ── retry 配置 ──────────────────────────────────────────

_retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
    retry=retry_if_exception(is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# ── 调用方的错误分发工具 ────────────────────────────────

ErrorHandler = Callable[[LLMException], None]
ErrorHandlers = dict[int, ErrorHandler]


def dispatch_error(exc: LLMException, handlers: ErrorHandlers | None):
    """按 status_code 分发到对应的处理器，支持 '*' 通配

    用法:
        dispatch_error(exc, {
            402: lambda e: send_dingtalk("余额不足"),
            '*':  lambda e: logger.error("未知错误: %s", e),
        })
    """
    if not handlers:
        return
    handler = handlers.get(exc.status_code) or handlers.get("*")
    if handler:
        try:
            handler(exc)
        except Exception:
            logger.exception("error handler failed")


# ── 内部辅助 ────────────────────────────────────────────

def _extract_status_code(exc: BaseException) -> int:
    if hasattr(exc, "status_code"):
        return exc.status_code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code
    return 0


# ── NormalModel ─────────────────────────────────────────

class NormalModel(ChatModel):
    def __init__(self):
        self._model = ChatOpenAI(
            api_key=os.getenv("MODEL_KEY"),
            model=os.getenv("MODEL_NAME"),
            base_url=os.getenv("MODEL_URL"),
            temperature=0,
            seed=42,
        )
        self._model_name = os.getenv("MODEL_NAME")

    def bind_tools(self, tools: list[BaseTool]):
        self._model = self._model.bind_tools(tools)

    @contextmanager
    def _translate(self):
        """将底层异常转为 LLMException 子类"""
        try:
            yield
        except LLMException:
            raise
        except Exception as e:
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise

    # ── 公开接口 ──────────────────────────────────────

    def invoke(self, messages: list[BaseMessage | str]) -> str:
        """单次调用，不重试"""
        with self._translate():
            return self._model.invoke(messages).content

    def stream(self, messages: list[BaseMessage | str]) -> Generator[AIMessageChunk, Any, None]:
        """单次流式，不重试"""
        with self._translate():
            yield from self._model.stream(messages)

    @_retry_decorator
    def _do_invoke(self, messages: list[BaseMessage | str]) -> str:
        with self._translate():
            return self._model.invoke(messages).content

    def invoke_with_retry(self, messages: list[BaseMessage | str]) -> str:
        """调用 LLM，自动重试可恢复错误"""
        try:
            return self._do_invoke(messages)
        except RetryError as e:
            cause = e.last_attempt.exception()  # type: ignore[union-attr]
            code = _extract_status_code(cause)
            raise LLMException(
                f"invoke failed after 3 retries: {cause}",
                status_code=code,
                model=self._model_name,
            ) from cause

