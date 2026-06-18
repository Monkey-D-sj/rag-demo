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
from rag.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """异步 embedding 模型，带重试与超时。"""

    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(
            api_key=settings.embedding_key,
            base_url=settings.embedding_url,
            timeout=30,
        )
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

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
