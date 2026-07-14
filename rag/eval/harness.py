import json
from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

from rag.config import Settings
from rag.db.postgres import create_pg_pool
from rag.document import store
from rag.document.retriever import KnowledgeRetriever, _lexical_query
from rag.eval import EVAL_KB_ID
from rag.eval.embeddings_cache import load_cache, resolve_embeddings, save_cache
from rag.eval.metrics import aggregate, evaluate_query
from rag.eval.style import phase, success
from rag.models.embedding import EmbeddingModel


@dataclass
class GoldenItem:
    id: str
    query: str
    gold_snippets: list[str]
    rewrite_query: str | None = None
    out_of_scope: bool = False
    category: str = ""


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
                    category=obj.get("category", ""),
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
    reranker=None,
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
) -> dict:
    """一次跑通检索链路，含可选 reranker 评测。

    fused / vec_only / bm25_only 使用改写后查询；
    当 golden 标注了 rewrite_query 时追加 raw / raw_vec / raw_bm25 原始查询对照；
    当 reranker 注入时追加 fused_reranked 完整链路。
    """
    print(phase("开始评测"))

    # ── 预计算所有 query embedding（缓存命中跳过 API 调用）──
    cache = load_cache()
    in_scope = [it for it in items if not it.out_of_scope]
    unique_queries = list(dict.fromkeys(
        (it.rewrite_query or it.query) for it in in_scope
    ))
    if any(it.rewrite_query for it in in_scope):
        unique_queries += list(dict.fromkeys(
            it.query for it in in_scope if it.rewrite_query
        ))
    unique_queries = list(dict.fromkeys(unique_queries))
    q_embs = await resolve_embeddings(embedding, unique_queries, cache)
    save_cache(cache)

    total = len(in_scope)
    fused_pq: list[dict] = []
    fused_rows_store: list[list[dict]] = []
    raw_pq: list[dict] = []
    vec_pq: list[dict] = []
    raw_vec_pq: list[dict] = []
    bm25_pq: list[dict] = []
    raw_bm25_pq: list[dict] = []

    print(f"\n评测中: {len(items)} 条 ({total} in-scope), 每条含 3+ 路检索…\n", flush=True)

    for idx, item in enumerate(in_scope, 1):
        rw = item.rewrite_query or item.query
        has_rewrite = item.rewrite_query is not None
        rw_emb = q_embs[rw]

        # ── fused：改写后 query 走检索管线 ──
        rows = await retriever.search(rw, None, top_k=top_k, query_emb=rw_emb)
        fused_pq.append({
            "id": item.id, "query": item.query,
            **evaluate_query([r["text"] for r in rows], item.gold_snippets, ks),
        })
        fused_rows_store.append(rows)

        # ── raw fused：原始 query 走完整管线 ──
        if has_rewrite:
            raw_emb = q_embs[item.query]
            raw_rows = await retriever.search(item.query, None, top_k=top_k, query_emb=raw_emb)
            raw_pq.append({
                "id": item.id, "query": item.query,
                **evaluate_query([r["text"] for r in raw_rows], item.gold_snippets, ks),
            })

        # ── vec_only：改写后向量召回 ──
        vec_rows = await store.search_chunks(pool, rw_emb, None, top_k)
        vec_pq.append(
            {"id": item.id, **evaluate_query([r["text"] for r in vec_rows], item.gold_snippets, ks)}
        )

        # ── raw vec：原始 query 向量召回 ──
        if has_rewrite:
            raw_emb = q_embs[item.query]
            raw_vec_rows = await store.search_chunks(pool, raw_emb, None, top_k)
            raw_vec_pq.append(
                {"id": item.id, **evaluate_query([r["text"] for r in raw_vec_rows], item.gold_snippets, ks)}
            )

        # ── bm25_only：改写后 BM25 ──
        lex = _lexical_query(rw)
        bm25_rows = (
            await store.search_chunks_bm25(pool, lex, None, top_k)
            if lex else []
        )
        bm25_pq.append(
            {"id": item.id, **evaluate_query([r["text"] for r in bm25_rows], item.gold_snippets, ks)}
        )

        # ── raw bm25：原始 query BM25 ──
        if has_rewrite:
            raw_lex = _lexical_query(item.query)
            raw_bm25_rows = (
                await store.search_chunks_bm25(pool, raw_lex, None, top_k)
                if raw_lex else []
            )
            raw_bm25_pq.append(
                {"id": item.id, **evaluate_query([r["text"] for r in raw_bm25_rows], item.gold_snippets, ks)}
            )

        # ── 进度 ──
        if idx % 10 == 0 or idx == 1 or idx == total:
            pct = idx * 100 // total
            print(f"  [{idx:>{len(str(total))}}/{total}] {pct:>3}% …", flush=True)

    print(success(f" 评测完成，共 {total} 条\n"))

    result = {
        "fused": {"aggregate": aggregate(fused_pq), "per_query": fused_pq},
        "vec_only": {"aggregate": aggregate(vec_pq), "per_query": vec_pq},
        "bm25_only": {"aggregate": aggregate(bm25_pq), "per_query": bm25_pq},
        "categories": {it.id: it.category for it in in_scope},
    }
    if raw_pq:
        result["raw"] = {"aggregate": aggregate(raw_pq), "per_query": raw_pq}
        result["raw_vec"] = {"aggregate": aggregate(raw_vec_pq), "per_query": raw_vec_pq}
        result["raw_bm25"] = {"aggregate": aggregate(raw_bm25_pq), "per_query": raw_bm25_pq}

    # ── fused_reranked：fused 结果经过 reranker 重排序 ──
    if reranker is not None:
        reranked_pq: list[dict] = []
        for item, rows in zip(in_scope, fused_rows_store):
            rw = item.rewrite_query or item.query
            reranked = await reranker.rerank(rw, rows, top_k=len(rows))
            reranked_pq.append({
                "id": item.id, "query": item.query,
                **evaluate_query([r["text"] for r in reranked], item.gold_snippets, ks),
            })
        result["fused_reranked"] = {"aggregate": aggregate(reranked_pq), "per_query": reranked_pq}
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
