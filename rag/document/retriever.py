import time

from rag.common.logging import get_logger
from rag.document import store
from rag.models.embedding import EmbeddingModel

logger = get_logger()


class KnowledgeRetriever:
    """知识库检索:embed 查询 → pgvector 相似度召回 document_chunks。"""

    def __init__(self, pool, embedding: EmbeddingModel) -> None:
        self._pool = pool
        self._embedding = embedding

    async def search(
        self, query: str, knowledge_base_id: str, top_k: int = 5
    ) -> list[dict]:
        t0 = time.perf_counter()
        query_embedding = (await self._embedding.embed([query]))[0]
        embed_ms = round((time.perf_counter() - t0) * 1000, 1)

        t1 = time.perf_counter()
        rows = await store.search_chunks(
            self._pool, query_embedding, knowledge_base_id, top_k
        )
        search_ms = round((time.perf_counter() - t1) * 1000, 1)

        fields = {
            "kb_id": knowledge_base_id,
            "query": query[:200],  # 截断,长文本进日志没有意义
            "top_k": top_k,
            "hits": [
                {
                    "chunk_id": str(r["id"]),
                    "chunk_index": r["chunk_index"],
                    "similarity": round(r["similarity"], 4),
                }
                for r in rows
            ],
            "embed_ms": embed_ms,
            "search_ms": search_ms,
        }
        if rows:
            logger.info("向量召回完成: %d 条", len(rows), extra=fields)
        else:
            # 空结果是检索质量最直接的信号,升级 WARNING
            logger.warning("向量召回为空", extra=fields)
        return rows
