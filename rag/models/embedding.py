import logging

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

logger = get_logger()


class EmbeddingModel:
    """异步 embedding 模型，带重试与超时。"""

    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(
            api_key=settings.EMBEDDING_KEY,
            base_url=settings.EMBEDDING_URL,
            timeout=30,
        )
        self._model = settings.EMBEDDING_MODEL
        self._dim = settings.EMBEDDING_DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
            retry=retry_if_exception(is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                rsp = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dim,
                    encoding_format="float",
                )
        ordered = sorted(rsp.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]
