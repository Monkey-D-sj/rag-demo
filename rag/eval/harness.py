import json
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from rag.config import Settings
from rag.db.postgres import create_pg_pool
from rag.document import store
from rag.document.retriever import KnowledgeRetriever, _lexical_query
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
    return pool, KnowledgeRetriever(pool, embedding, settings)


async def run_eval(
    items: list[GoldenItem],
    pool: AsyncConnectionPool,
    embedding: EmbeddingModel,
    retriever: KnowledgeRetriever,
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
) -> dict:
    """一次跑通三条检索链路：fused（生产路径）、vec_only、bm25_only。

    每条 query 共享一次 embedding，分别评测三条路的指标。返回：
    {"fused": {aggregate, per_query}, "vec_only": {aggregate}, "bm25_only": {aggregate}}
    """
    fused_pq: list[dict] = []
    vec_pq: list[dict] = []
    bm25_pq: list[dict] = []

    for item in items:
        # ── fused：走生产 KnowledgeRetriever（merge + reranker） ──
        rows = await retriever.search(item.query, EVAL_KB_ID, top_k=top_k)
        fused_pq.append({
            "id": item.id, "query": item.query,
            **evaluate_query([r["text"] for r in rows], item.gold_snippets, ks),
        })

        # ── vec_only：原始向量召回，共享一次 embedding ──
        emb = (await embedding.embed([item.query]))[0]
        vec_rows = await store.search_chunks(pool, emb, EVAL_KB_ID, top_k)
        vec_pq.append(
            evaluate_query([r["text"] for r in vec_rows], item.gold_snippets, ks)
        )

        # ── bm25_only：原始 BM25 召回 ──
        lex = _lexical_query(item.query)
        bm25_rows = (
            await store.search_chunks_bm25(pool, lex, EVAL_KB_ID, top_k)
            if lex else []
        )
        bm25_pq.append(
            evaluate_query([r["text"] for r in bm25_rows], item.gold_snippets, ks)
        )

    return {
        "fused": {"aggregate": aggregate(fused_pq), "per_query": fused_pq},
        "vec_only": {"aggregate": aggregate(vec_pq)},
        "bm25_only": {"aggregate": aggregate(bm25_pq)},
    }
