import asyncio
import time

from rag.common.logging import get_logger
from rag.document import store
from rag.models.embedding import EmbeddingModel
from rag.observability.langfuse import observe_if_enabled

logger = get_logger()

# RRF 常数与每路候选倍数:检索内部细节,有评测数据前不进 Settings。
RRF_K = 60
CANDIDATE_MULTIPLIER = 2


def _rrf_fuse(
    vec_rows: list[dict], bm25_rows: list[dict], top_k: int
) -> list[dict]:
    """RRF 融合:score = Σ 1/(RRF_K + rank),按 id 去重,行数据取先见者。"""
    fused: dict[str, dict] = {}
    for source, rows in (("vec", vec_rows), ("bm25", bm25_rows)):
        for rank, row in enumerate(rows):
            entry = fused.setdefault(
                str(row["id"]), {"row": row, "rrf_score": 0.0, "sources": []}
            )
            entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            entry["sources"].append(source)
    ranked = sorted(
        fused.values(), key=lambda e: e["rrf_score"], reverse=True
    )[:top_k]
    results = []
    for entry in ranked:
        row = dict(entry["row"])
        row["rrf_score"] = round(entry["rrf_score"], 6)
        row["sources"] = entry["sources"]
        results.append(row)
    return results


def _rank_summary(rows: list[dict], score_key: str) -> list[dict]:
    """单路排名摘要进日志:排查「为什么这条被召回」时看各路贡献。"""
    return [
        {
            "chunk_id": str(r["id"]),
            "chunk_index": r["chunk_index"],
            score_key: round(r[score_key], 4),
            "text_preview": r["text"][:80],
        }
        for r in rows
    ]


class KnowledgeRetriever:
    """知识库检索:向量(pgvector)+BM25(pg_search/jieba)并发召回,RRF 融合。"""

    def __init__(self, pool, embedding: EmbeddingModel) -> None:
        self._pool = pool
        self._embedding = embedding

    @observe_if_enabled(name="knowledge_retrieve")
    async def search(
        self, query: str, knowledge_base_id: str, top_k: int = 5
    ) -> list[dict]:
        candidates = top_k * CANDIDATE_MULTIPLIER
        timings: dict[str, float] = {}

        async def _vec_leg() -> list[dict]:
            t0 = time.perf_counter()
            emb = (await self._embedding.embed([query]))[0]
            timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            t1 = time.perf_counter()
            rows = await store.search_chunks(
                self._pool, emb, knowledge_base_id, candidates
            )
            timings["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            return rows

        async def _bm25_leg() -> list[dict]:
            t0 = time.perf_counter()
            try:
                rows = await store.search_chunks_bm25(
                    self._pool, query, knowledge_base_id, candidates
                )
            except Exception:  # noqa: BLE001 - 词法路失败降级,不拖垮检索
                logger.warning("BM25 召回失败,降级纯向量", exc_info=True)
                rows = []
            timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return rows

        # 向量路异常照常传播(主路径语义不变);BM25 路已在内部兜底
        vec_rows, bm25_rows = await asyncio.gather(_vec_leg(), _bm25_leg())
        results = _rrf_fuse(vec_rows, bm25_rows, top_k)

        fields = {
            "kb_id": knowledge_base_id,
            "query": query[:200],  # 截断,长文本进日志没有意义
            "top_k": top_k,
            "vec_hits": _rank_summary(vec_rows, "similarity"),
            "bm25_hits": _rank_summary(bm25_rows, "score"),
            "hits": [
                {
                    "chunk_id": str(r["id"]),
                    "chunk_index": r["chunk_index"],
                    "rrf_score": r["rrf_score"],
                    "sources": r["sources"],
                    "text_preview": r["text"][:80],
                }
                for r in results
            ],
            **timings,
        }
        if results:
            logger.info(
                "混合召回完成: %d 条 (向量 %d + BM25 %d)",
                len(results), len(vec_rows), len(bm25_rows),
                extra=fields,
            )
        else:
            # 空结果是检索质量最直接的信号,升级 WARNING
            logger.warning("混合召回为空", extra=fields)
        return results
