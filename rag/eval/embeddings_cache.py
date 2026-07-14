"""Query embedding 缓存，避免每次评测重复 embed 相同 query。

缓存文件为 datasets/query_embeddings.json，key 格式 "{model}:{query}"，
换 embedding 模型后自动失效（key 不匹配），无需手动清理。
"""

import json
from pathlib import Path

from rag.eval import DATASETS_DIR
from rag.models.embedding import EmbeddingModel

CACHE_PATH = DATASETS_DIR / "query_embeddings.json"


def _cache_key(model_name: str, query: str) -> str:
    return f"{model_name}:{query}"


def load_cache() -> dict[str, list[float]]:
    """读取缓存；文件不存在返回空 dict。"""
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict[str, list[float]]) -> None:
    """写入缓存文件。"""
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


async def resolve_embeddings(
    embedding: EmbeddingModel,
    queries: list[str],
    cache: dict[str, list[float]],
) -> dict[str, list[float]]:
    """批量解析 query embedding：缓存命中直接用，未命中调用 API 后写入缓存。

    Returns: {query: embedding_vector}。
    """
    model = embedding.model
    result: dict[str, list[float]] = {}
    missing: list[str] = []

    for q in queries:
        key = _cache_key(model, q)
        cached = cache.get(key)
        if cached is not None and len(cached) == embedding.dim:
            result[q] = cached
        else:
            missing.append(q)

    if missing:
        # 去重后批量 embed（API 限制单批 ≤10）
        unique = list(dict.fromkeys(missing))
        vectors = await embedding.embed(unique)
        for q, vec in zip(unique, vectors):
            result[q] = vec
            cache[_cache_key(model, q)] = vec

    return result
