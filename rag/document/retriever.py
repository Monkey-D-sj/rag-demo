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
    vec_rows: list[dict], bm25_rows: list[dict], top_k: int
) -> list[dict]:
    """合并两路召回结果：去重（vec 优先），保留原始分，截断 top_k。

    最终排名由下游 reranker 负责，此处不做融合排序。
    """
    seen: set[str] = set()
    results: list[dict] = []

    # vec 路优先（已过相似度阈值过滤）
    for row in vec_rows:
        rid = str(row["id"])
        if rid not in seen:
            seen.add(rid)
            r = dict(row)
            r["sources"] = ["vec"]
            r["score"] = None
            results.append(r)

    # bm25 路补充
    for row in bm25_rows:
        rid = str(row["id"])
        if rid not in seen:
            seen.add(rid)
            r = dict(row)
            r["sources"] = ["bm25"]
            r["similarity"] = None
            results.append(r)
        else:
            # 两路都命中：标记 sources，补充 bm25 原始分
            for r in results:
                if str(r["id"]) == rid:
                    r["sources"].append("bm25")
                    r["score"] = row.get("score")
                    break

    return results[:top_k]


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

    @observe_if_enabled(name="knowledge_retrieve")
    async def search(
        self, query: str, knowledge_base_ids: list[str] | None = None, top_k: int = 5
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
        results = _merge_dedup(vec_rows, bm25_rows, top_k)

        return results
