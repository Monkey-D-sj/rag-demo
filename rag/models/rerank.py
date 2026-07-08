from pydantic import BaseModel, Field

from rag.common.logging import get_logger
from rag.config import Settings
from rag.models.base import ChatModel

logger = get_logger()


# ── LLM Reranker ──

class _ChunkScore(BaseModel):
    chunk_id: str = Field(description="chunk 的 id")
    score: float = Field(ge=0, le=10, description="0-10 相关性分数")


class _RerankResult(BaseModel):
    scores: list[_ChunkScore]


# ── DashScope Qwen3 Reranker ──


class QwenReranker:
    """DashScope qwen3-rerank 重排序器，直接走 HTTP，不依赖 SDK。

    RERANK_BASE_URL 示例:
      https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-api
    """

    def __init__(self, settings: Settings):
        import httpx

        if not settings.RERANK_BASE_URL:
            raise ValueError("RERANK_BASE_URL 未配置，无法初始化 QwenReranker")
        self._client = httpx.AsyncClient(
            base_url=settings.RERANK_BASE_URL.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.RERANK_KEY or settings.EMBEDDING_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        self._model = settings.RERANK_MODEL

    async def rerank(
        self, query: str, chunks: list[dict], top_k: int | None = None
    ) -> list[dict]:
        if not chunks:
            return []

        documents = [str(c.get("text", "")) for c in chunks]
        body = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(chunks),
        }

        try:
            rsp = await self._client.post("/v1/reranks", json=body)
            rsp.raise_for_status()
            data = rsp.json()
        except Exception:
            logger.warning("QwenRerank API 调用失败，降级为原始召回顺序", exc_info=True)
            return chunks

        # {"results": [{"index": int, "relevance_score": float}, ...]}
        results = data.get("results", [])
        if not results:
            return chunks

        score_map = {r["index"]: r["relevance_score"] for r in results}
        indexed = sorted(
            enumerate(chunks),
            key=lambda t: score_map.get(t[0], 0),
            reverse=True,
        )
        for idx, c in indexed:
            c["rerank_score"] = score_map.get(idx, 0)

        result = [c for _, c in indexed]
        return result[:top_k] if top_k is not None else result


# ── shared ──

def _items_text(items: list[dict]) -> str:
    """将候选列表格式化为 LLM 可读的文本。"""
    lines = []
    for i, item in enumerate(items, 1):
        lines.append(f"[{i}] id={item['chunk_id']}\n{item['text']}")
    return "\n\n".join(lines)
