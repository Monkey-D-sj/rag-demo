import asyncio
import re
import time

from rag.common.logging import get_logger
from rag.document import store
from rag.models.embedding import EmbeddingModel
from rag.observability.langfuse import observe_if_enabled, span_scope

logger = get_logger()

# RRF 常数与每路候选倍数:检索内部细节,有评测数据前不进 Settings。
RRF_K = 60
CANDIDATE_MULTIPLIER = 2
# 向量路相似度阈值:低于此值的 chunk 不进 RRF 融合,避免低分噪音稀释排名。
VEC_SIMILARITY_THRESHOLD = 0.5


def _lexical_query(query: str) -> str:
    """BM25 路输入卫生:剥掉标点符号。纯符号查询在索引里能匹配数千个近零分
    chunk,而 RRF 只看排名不看分数,会让噪音与向量命中 1:1 交错挤进 top_k。"""
    return re.sub(r"[\W_]+", " ", query).strip()


# vec 路源优先级高于 bm25:同 id chunk 在两路都命中时取 vec 行的字段。
_VEC_SOURCE_PRIORITY = 0
_BM25_SOURCE_PRIORITY = 1


def _rrf_fuse(
    vec_rows: list[dict], bm25_rows: list[dict], top_k: int
) -> list[dict]:
    """RRF 融合:score = Σ 1/(RRF_K + rank),按 id 去重。

    行数据取高优先级路（vec > bm25），但两路各自原始分在两路都命中时均保留。
    """
    fused: dict[str, dict] = {}
    for source, priority, rows in (
        ("vec", _VEC_SOURCE_PRIORITY, vec_rows),
        ("bm25", _BM25_SOURCE_PRIORITY, bm25_rows),
    ):
        for rank, row in enumerate(rows):
            rid = str(row["id"])
            entry = fused.get(rid)
            if entry is None:
                fused[rid] = {
                    "row": row,
                    "rrf_score": 0.0,
                    "sources": [],
                    "_priority": priority,
                    # 两路原始分独立保存，避免被高优先级行数据覆盖丢失
                    "_vec_similarity": None,
                    "_bm25_score": None,
                }
                entry = fused[rid]
            elif priority < entry["_priority"]:
                entry["row"] = row
                entry["_priority"] = priority
            # 按来源路捕获原始分，无论是否第一次命中
            if source == "vec" and "similarity" in row:
                entry["_vec_similarity"] = row["similarity"]
            elif source == "bm25" and "score" in row:
                entry["_bm25_score"] = row["score"]
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
        row["similarity"] = entry["_vec_similarity"]
        row["score"] = entry["_bm25_score"]
        results.append(row)
    return results


def _rank_summary(rows: list[dict], score_key: str) -> list[dict]:
    """单路排名摘要进日志:排查「为什么这条被召回」时看各路贡献。"""
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
        lexical_query = _lexical_query(query)

        span_input = {"query": query[:200], "kb_id": knowledge_base_id, "candidates": candidates}

        async def _vec_leg() -> list[dict]:
            # 子 span 让两路在 trace 里各自可见,而不是只有融合后的黑盒输出
            with span_scope("vector_recall", input=span_input) as span:
                t0 = time.perf_counter()
                try:
                    emb = (await self._embedding.embed([query]))[0]
                    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    t1 = time.perf_counter()
                    rows = await store.search_chunks(
                        self._pool, emb, knowledge_base_id, candidates
                    )
                    timings["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                    # 低相似度截断:低于阈值的 chunk 不进 RRF,减少噪音稀释
                    before = len(rows)
                    rows = [r for r in rows if r.get("similarity", 0) >= VEC_SIMILARITY_THRESHOLD]
                    if before > len(rows):
                        logger.debug(
                            "向量路截断 %d/%d 条 (threshold=%.2f)",
                            before - len(rows), before, VEC_SIMILARITY_THRESHOLD,
                        )
                    if span is not None:
                        span.update(output=_rank_summary(rows, "similarity"))
                    return rows
                except Exception:  # noqa: BLE001 - 向量路失败降级,与 BM25 路对等容错
                    logger.warning("向量召回失败,降级纯 BM25", exc_info=True)
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
                            self._pool, lexical_query, knowledge_base_id, candidates
                        )
                    except Exception:  # noqa: BLE001 - 词法路失败降级,不拖垮检索
                        logger.warning("BM25 召回失败,降级纯向量", exc_info=True)
                        rows = []
                timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                if span is not None:
                    span.update(output=_rank_summary(rows, "score"))
                return rows

        # 两路并发,各自内部兜底;单路失败降级不影响另一路,双路皆败返回空。
        vec_rows, bm25_rows = await asyncio.gather(_vec_leg(), _bm25_leg())
        results = _rrf_fuse(vec_rows, bm25_rows, top_k)

        return results
