import json
from dataclasses import dataclass
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from rag.config import Settings
from rag.db.postgres import create_pg_pool
from rag.document.retriever import KnowledgeRetriever
from rag.eval import EVAL_KB_ID
from rag.eval.metrics import aggregate, evaluate_query
from rag.models.embedding import EmbeddingModel


@dataclass
class GoldenItem:
    id: str
    query: str
    gold_snippets: list[str]


def load_golden(path) -> list[GoldenItem]:
    """读取 jsonl golden 集；跳过空行；校验必填字段，缺失即报错。"""
    items: list[GoldenItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("query") or not obj.get("gold_snippets"):
                raise ValueError(f"{path}:{lineno} 缺少 query 或 gold_snippets")
            items.append(
                GoldenItem(
                    id=str(obj.get("id", lineno)),
                    query=obj["query"],
                    gold_snippets=list(obj["gold_snippets"]),
                )
            )
    return items


async def build_retriever(settings: Settings) -> tuple[AsyncConnectionPool, KnowledgeRetriever]:
    """构造 pool + 复用生产 KnowledgeRetriever；调用方负责 pool.close()。"""
    pool = await create_pg_pool(settings)
    embedding = EmbeddingModel(settings)
    return pool, KnowledgeRetriever(pool, embedding)


async def run_eval(
    items: list[GoldenItem],
    retriever: KnowledgeRetriever,
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
) -> dict:
    """对每条 golden 跑检索并算指标，返回 {"aggregate", "per_query"}。"""
    per_query: list[dict] = []
    for item in items:
        rows = await retriever.search(item.query, EVAL_KB_ID, top_k=top_k)
        texts = [r["text"] for r in rows]
        metrics = evaluate_query(texts, item.gold_snippets, ks)
        per_query.append({"id": item.id, "query": item.query, **metrics})

    metric_keys = [k for k in per_query[0] if k not in ("id", "query")] if per_query else []
    agg = aggregate([{k: q[k] for k in metric_keys} for q in per_query])
    return {"aggregate": agg, "per_query": per_query}
