from __future__ import annotations

from collections import defaultdict

from rag.common.logging import get_logger
from rag.document import store

logger = get_logger()

# 精确匹配种子实体,沿 RELATES 扩 1 跳收集邻居实体的 chunk_ids
_EXPAND_QUERY = """
UNWIND $names AS name
MATCH (e:Entity {name: name})
OPTIONAL MATCH (e)-[:RELATES]-(nb:Entity)
RETURN e.name AS seed,
       e.chunk_ids AS seed_chunks,
       collect(nb.chunk_ids) AS neighbor_chunk_lists
"""

_SEED_WEIGHT = 2
_NEIGHBOR_WEIGHT = 1


def _parse_uid(uid: str) -> tuple[str, int] | None:
    """chunk_uid 格式 "{document_id}:{chunk_index}";异常格式返回 None 跳过。"""
    doc_id, sep, idx = uid.rpartition(":")
    if not sep or not doc_id or not idx.isdigit():
        return None
    return doc_id, int(idx)


def _score_chunks(records: list[dict]) -> list[tuple[str, int]]:
    """图命中 chunk 加权评分:种子实体 chunk 权重 2,邻居权重 1,多实体命中累加。

    返回按分数降序(同分保持首次出现序)的 (document_id, chunk_index) 列表。
    """
    scores: dict[tuple[str, int], int] = defaultdict(int)
    order: dict[tuple[str, int], int] = {}

    def _add(uid: str, weight: int) -> None:
        parsed = _parse_uid(uid)
        if parsed is None:
            return
        scores[parsed] += weight
        order.setdefault(parsed, len(order))

    for rec in records:
        for uid in rec.get("seed_chunks") or []:
            _add(uid, _SEED_WEIGHT)
        for chunk_list in rec.get("neighbor_chunk_lists") or []:
            for uid in chunk_list or []:
                _add(uid, _NEIGHBOR_WEIGHT)

    return sorted(scores, key=lambda k: (-scores[k], order[k]))


class GraphRetriever:
    """图召回:查询实体 1 跳扩展 → chunk_uid 评分排序 → 回 pg 取正文。

    异常向上抛,由 KnowledgeRetriever 的图路协程统一降级。
    """

    def __init__(self, driver, database: str, pool) -> None:
        self._driver = driver
        self._database = database
        self._pool = pool

    async def search(self, entities: list[str], limit: int) -> list[dict]:
        if not entities:
            return []

        async with self._driver.session(database=self._database) as session:
            result = await session.run(_EXPAND_QUERY, names=entities)
            records = await result.data()

        logger.info("图召回: 查询实体 %d 个,命中种子实体 %d 个", len(entities), len(records))

        ranked = _score_chunks(records)[:limit]
        if not ranked:
            return []

        rows = await store.get_chunks_by_uids(self._pool, ranked)
        # SQL 返回顺序不保证,按图评分序重排
        pos = {uid: i for i, uid in enumerate(ranked)}
        rows.sort(key=lambda r: pos.get((str(r["document_id"]), r["chunk_index"]), len(pos)))
        return rows
