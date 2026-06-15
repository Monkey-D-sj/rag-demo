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
    retry_if_exception,
    before_sleep_log,
    RetryError,
)

from rag.common.exception import (
    LLMException,
    LLMNonRetryableError,
    from_http_error,
    is_retryable,
)
from rag.models.base import ChatModel

load_dotenv()

logger = logging.getLogger(__name__)

_retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
    retry=retry_if_exception(is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _extract_status_code(exc: BaseException) -> int:
    """从 OpenAI / httpx 异常中提取 HTTP 状态码"""
    if hasattr(exc, "status_code"):
        return exc.status_code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code
    return 0


class NormalModel(ChatModel):
    def __init__(self):
        self._model = ChatOpenAI(
            api_key=os.getenv("MODEL_KEY"),
            model=os.getenv("MODEL_NAME"),
            base_url=os.getenv("MODEL_URL"),
            temperature=0,
            seed=42,
        )

    @property
    def _model_name(self) -> str:
        return getattr(self._model, "model_name", "")

    def bind_tools(self, tools: list[BaseTool]):
        self._model = self._model.bind_tools(tools)

    @_retry_decorator
    def _do_invoke(self, messages: list[Any | str]) -> str:
        """带重试的 invoke；不可重试异常会立即穿透"""
        try:
            return self._model.invoke(messages).content
        except Exception as e:
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise  # 未知异常交给 is_retryable 判断

    # ── 公开接口 ──────────────────────────────────────────

    def invoke(self, messages: list[Any | str]) -> str:
        """单次调用 LLM，不重试"""
        try:
            return self._model.invoke(messages).content
        except Exception as e:
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise

    def stream(self, messages: list[Any | str]) -> Generator[AIMessageChunk, Any, None]:
        """单次流式调用 LLM，不重试"""
        try:
            yield from self._model.stream(messages)
        except Exception as e:
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise

    def invoke_with_retry(self, messages: list[Any | str]) -> str:
        """调用 LLM，不可重试错误立即失败，可重试错误自动重试"""
        try:
            return self._do_invoke(messages)
        except LLMNonRetryableError:
            raise  # 不可重试，直接抛
        except RetryError as e:
            cause: BaseException = e.last_attempt.exception()  # type: ignore[union-attr]
            code = _extract_status_code(cause)
            raise LLMException(
                f"invoke failed after 3 retries: {cause}",
                status_code=code,
                model=self._model_name,
                retryable=False,
            ) from cause

    def stream_with_retry(
        self, messages: list[Any | str]
    ) -> Generator[AIMessageChunk | str, Any, None]:
        """流式调用 LLM，失败时降级到 invoke_with_retry 兜底"""
        try:
            yield from self._model.stream(messages)
            return
        except LLMNonRetryableError:
            raise
        except Exception as e:
            code = _extract_status_code(e)
            if code and not is_retryable(from_http_error(code, str(e))):
                raise from_http_error(code, str(e), model=self._model_name) from e
            logger.warning("stream interrupted, falling back to invoke: %s", e)

        yield AIMessageChunk(content="\n\n[响应中断，正在恢复...]\n\n")
        try:
            result = self._do_invoke(messages)
            yield AIMessageChunk(content=result)
        except LLMNonRetryableError:
            raise
        except RetryError as e:
            cause = e.last_attempt.exception()  # type: ignore[union-attr]
            code = _extract_status_code(cause)
            raise LLMException(
                f"invoke fallback failed after 3 retries: {cause}",
                status_code=code,
                model=self._model_name,
                retryable=False,
            ) from cause
