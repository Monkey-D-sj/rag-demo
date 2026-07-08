import json
from dataclasses import dataclass, field

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
    rewrite_query: str | None = None
    out_of_scope: bool = False


def load_golden(path) -> list[GoldenItem]:
    """读取 jsonl golden 集；跳过空行；校验必填字段，缺失即报错。

    可选字段：rewrite_query（改写后查询）、out_of_scope（范围外，默认 false）。
    范围外条目允许 gold_snippets 为空。
    """
    items: list[GoldenItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("query"):
                raise ValueError(f"{path}:{lineno} 缺少 query")
            out_of_scope = bool(obj.get("out_of_scope", False))
            if not out_of_scope and not obj.get("gold_snippets"):
                raise ValueError(f"{path}:{lineno} 缺少 gold_snippets（非 out_of_scope 条目必填）")
            items.append(
                GoldenItem(
                    id=str(obj.get("id", lineno)),
                    query=obj["query"],
                    gold_snippets=list(obj.get("gold_snippets", [])),
                    rewrite_query=obj.get("rewrite_query"),
                    out_of_scope=out_of_scope,
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
    """一次跑通四条检索链路：fused、raw（原始查询）、vec_only、bm25_only。

    只评测 in-scope 条目（out_of_scope 条目不参与检索评测）。
    返回 {"fused": {aggregate, per_query}, "raw": {...}, "vec_only": {...}, "bm25_only": {...}}
    """
    fused_pq: list[dict] = []
    raw_pq: list[dict] = []
    vec_pq: list[dict] = []
    bm25_pq: list[dict] = []

    in_scope = [it for it in items if not it.out_of_scope]

    for item in in_scope:
        # ── fused：用 rewrite_query（生产路径：查询改写后的检索） ──
        search_query = item.rewrite_query or item.query
        rows = await retriever.search(search_query, EVAL_KB_ID, top_k=top_k)
        fused_pq.append({
            "id": item.id, "query": item.query,
            **evaluate_query([r["text"] for r in rows], item.gold_snippets, ks),
        })

        # ── raw：用原始 query（跳过查询改写） ──
        if item.rewrite_query:
            raw_rows = await retriever.search(item.query, EVAL_KB_ID, top_k=top_k)
            raw_pq.append({
                "id": item.id, "query": item.query,
                **evaluate_query([r["text"] for r in raw_rows], item.gold_snippets, ks),
            })

        # ── vec_only：原始向量召回，共享一次 embedding ──
        emb = (await embedding.embed([search_query]))[0]
        vec_rows = await store.search_chunks(pool, emb, EVAL_KB_ID, top_k)
        vec_pq.append(
            evaluate_query([r["text"] for r in vec_rows], item.gold_snippets, ks)
        )

        # ── bm25_only：原始 BM25 召回 ──
        lex = _lexical_query(search_query)
        bm25_rows = (
            await store.search_chunks_bm25(pool, lex, EVAL_KB_ID, top_k)
            if lex else []
        )
        bm25_pq.append(
            evaluate_query([r["text"] for r in bm25_rows], item.gold_snippets, ks)
        )

    result = {
        "fused": {"aggregate": aggregate(fused_pq), "per_query": fused_pq},
        "vec_only": {"aggregate": aggregate(vec_pq)},
        "bm25_only": {"aggregate": aggregate(bm25_pq)},
    }
    if raw_pq:
        result["raw"] = {"aggregate": aggregate(raw_pq)}
    return result


def eval_out_of_scope(
    items: list[GoldenItem],
) -> dict:
    """评测范围判断分类准确率（不调 LLM，仅统计 golden 标注）。

    返回 {"total": N, "in_scope": N, "out_of_scope": N}。
    实际 LLM 分类评测由 rag-eval 内部调 handle_query 完成。
    """
    total = len(items)
    in_scope = sum(1 for it in items if not it.out_of_scope)
    out_of_scope = total - in_scope
    return {"total": total, "in_scope": in_scope, "out_of_scope": out_of_scope}
