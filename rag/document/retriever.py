import asyncio
import re
import time

from rag.common.logging import get_logger
from rag.config import Settings
from rag.document import store
from rag.models.embedding import EmbeddingModel
from rag.observability.langfuse import observe_if_enabled, span_scope

logger = get_logger()


def _lexical_query(query: str) -> str:
    """BM25 路输入卫生：剥掉标点符号，避免纯符号查询匹配大量近零分噪音。"""
    return re.sub(r"[\W_]+", " ", query).strip()


def _merge_dedup(
    vec_rows: list[dict], bm25_rows: list[dict], top_k: int, rrf_k: int = 60
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 融合两路召回结果。

    对每路按排名计算 RRF 分: 1/(k + rank)，两路分数加和，
    按总分降序排列取 top_k。同一 chunk 在两路均命中时累加 RRF 分。
    """
    if not vec_rows and not bm25_rows:
        return []

    chunk_map: dict[str, dict] = {}

    # vec 路：rank 1 = 最高相似度
    for rank, row in enumerate(vec_rows, 1):
        rid = str(row["id"])
        rrf = 1.0 / (rrf_k + rank)
        if rid in chunk_map:
            chunk_map[rid]["sources"].append("vec")
            chunk_map[rid]["rrf_score"] = chunk_map[rid]["rrf_score"] + rrf
        else:
            r = dict(row)
            r["sources"] = ["vec"]
            r["rrf_score"] = rrf
            r["score"] = None
            chunk_map[rid] = r

    # bm25 路：rank 1 = 最高分
    for rank, row in enumerate(bm25_rows, 1):
        rid = str(row["id"])
        rrf = 1.0 / (rrf_k + rank)
        if rid in chunk_map:
            chunk_map[rid]["sources"].append("bm25")
            chunk_map[rid]["rrf_score"] = chunk_map[rid]["rrf_score"] + rrf
            chunk_map[rid]["score"] = row.get("score")
        else:
            r = dict(row)
            r["sources"] = ["bm25"]
            r["rrf_score"] = rrf
            r["similarity"] = None
            chunk_map[rid] = r

    # 按 RRF 分降序，同分时保持插入顺序（vec 先插入的优先）
    sorted_chunks = sorted(
        chunk_map.values(),
        key=lambda r: r["rrf_score"],
        reverse=True,
    )

    return sorted_chunks[:top_k]


def _rank_summary(rows: list[dict], score_key: str) -> list[dict]:
    """单路排名摘要进日志：排查「为什么这条被召回」时看各路贡献。"""
    return [
        {
            "chunk_id": str(r["id"]),
            "chunk_index": r["chunk_index"],
            score_key: round(r[score_key], 4),
            "text": r["text"],
        }
        for r in rows
    ]


class KnowledgeRetriever:
    """知识库检索：向量(pgvector)+BM25(pg_search/jieba)并发召回，去重合并后由 reranker 重排。"""

    def __init__(self, pool, embedding: EmbeddingModel, settings: Settings) -> None:
        self._pool = pool
        self._embedding = embedding
        self._candidate_multiplier = settings.RETRIEVER_CANDIDATE_MULTIPLIER
        self._vec_threshold = settings.RETRIEVER_VEC_SIMILARITY_THRESHOLD
        self._rrf_k = settings.RETRIEVER_RRF_K

    @observe_if_enabled(name="knowledge_retrieve")
    async def search(
        self, query: str, knowledge_base_ids: list[str] | None = None, top_k: int = 5,
        query_emb: list[float] | None = None,
    ) -> list[dict]:
        candidates = top_k * self._candidate_multiplier
        timings: dict[str, float] = {}
        lexical_query = _lexical_query(query)

        span_input = {"query": query[:200], "kb_ids": knowledge_base_ids, "candidates": candidates}

        async def _vec_leg() -> list[dict]:
            # 子 span 让两路在 trace 里各自可见，而不是只有融合后的黑盒输出
            with span_scope("vector_recall", input=span_input) as span:
                t0 = time.perf_counter()
                try:
                    if query_emb is not None:
                        emb = query_emb
                        timings["embed_ms"] = 0
                    else:
                        emb = (await self._embedding.embed([query]))[0]
                        timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    t1 = time.perf_counter()
                    rows = await store.search_chunks(
                        self._pool, emb, knowledge_base_ids, candidates
                    )
                    timings["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                    # 低相似度截断：低于阈值的 chunk 不参与合并，减少噪音稀释
                    before = len(rows)
                    threshold = self._vec_threshold
                    rows = [r for r in rows if r.get("similarity", 0) >= threshold]
                    if before > len(rows):
                        logger.debug(
                            "向量路截断 %d/%d 条 (threshold=%.2f)",
                            before - len(rows), before, threshold,
                        )
                    if span is not None:
                        span.update(output=_rank_summary(rows, "similarity"))
                    return rows
                except Exception:  # noqa: BLE001 - 向量路失败降级，与 BM25 路对等容错
                    logger.warning("向量召回失败，降级纯 BM25", exc_info=True)
                    timings.setdefault("embed_ms", 0)
                    timings.setdefault("search_ms", 0)
                    return []

        async def _bm25_leg() -> list[dict]:
            with span_scope("bm25_recall", input=span_input) as span:
                t0 = time.perf_counter()
                if not lexical_query:
                    rows: list[dict] = []
                else:
                    try:
                        rows = await store.search_chunks_bm25(
                            self._pool, lexical_query, knowledge_base_ids, candidates
                        )
                    except Exception:  # noqa: BLE001 - 词法路失败降级，不拖垮检索
                        logger.warning("BM25 召回失败，降级纯向量", exc_info=True)
                        rows = []
                timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                if span is not None:
                    span.update(output=_rank_summary(rows, "score"))
                return rows

        # 两路并发，各自内部兜底；单路失败降级不影响另一路，双路皆败返回空。
        vec_rows, bm25_rows = await asyncio.gather(_vec_leg(), _bm25_leg())
        results = _merge_dedup(vec_rows, bm25_rows, top_k, self._rrf_k)

        return results

    async def fetch_parent_contents(self, document_ids: list[str]) -> dict[str, str]:
        """按 doc_id 批量补查父文档全文（parent-child retrieval 展开用）。"""
        return await store.get_documents_content(self._pool, document_ids)
