from rag.document import store
from rag.models.embedding import EmbeddingModel


class KnowledgeRetriever:
    """知识库检索:embed 查询 → pgvector 相似度召回 document_chunks。"""

    def __init__(self, pool, embedding: EmbeddingModel) -> None:
        self._pool = pool
        self._embedding = embedding

    async def search(
        self, query: str, knowledge_base_id: str, top_k: int = 5
    ) -> list[dict]:
        query_embedding = (await self._embedding.embed([query]))[0]
        return await store.search_chunks(
            self._pool, query_embedding, knowledge_base_id, top_k
        )
