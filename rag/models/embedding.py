import logging
from contextlib import nullcontext

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.common.exception import is_retryable
from rag.common.logging import get_logger
from rag.config import Settings
from rag.governance.config import get_governance_settings
from rag.governance.guard import LLMGuard

logger = get_logger()


class EmbeddingModel:
    """异步 embedding 模型,带重试与超时;guard 注入后纳入限流/熔断/统计。"""

    def __init__(self, settings: Settings, guard: LLMGuard | None = None):
        gov = get_governance_settings()
        self._client = AsyncOpenAI(
            api_key=settings.EMBEDDING_KEY,
            base_url=settings.EMBEDDING_URL,
            timeout=gov.EMBEDDING_TIMEOUT_SECONDS,
        )
        self._model = settings.EMBEDDING_MODEL
        self._dim = settings.EMBEDDING_DIM
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._guard = guard

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _acquire(self):
        if self._guard is None:
            return nullcontext()
        return self._guard.acquire("embedding")

    def _track(self):
        if self._guard is None:
            return nullcontext()
        return self._guard.track("embedding", self._model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self._track() as tracker:
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
                    async with self._acquire():
                        rsp = await self._client.embeddings.create(
                            model=self._model,
                            input=texts,
                            dimensions=self._dim,
                            encoding_format="float",
                        )
            usage = getattr(rsp, "usage", None)
            if tracker is not None and usage is not None:
                # embedding 只有输入侧 token
                tracker.set_tokens(getattr(usage, "prompt_tokens", None), None)
        ordered = sorted(rsp.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]
