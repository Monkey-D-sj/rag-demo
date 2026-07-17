import logging

import pytest

from rag.config import Settings
from rag.document.retriever import KnowledgeRetriever, _merge_dedup


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1, 0.2]]


def _settings() -> Settings:
    return Settings()


def _row(cid: str, text: str = "正文", **extra) -> dict:
    return {
        "id": cid,
        "document_id": "d1",
        "chunk_index": 0,
        "text": text,
        **extra,
    }


def _retriever(monkeypatch, vec_rows, bm25_rows=None, bm25_exc=None):
    import rag.document.retriever as mod

    async def fake_vec(pool, embedding, knowledge_base_ids, top_k):
        return vec_rows

    async def fake_bm25(pool, query_text, knowledge_base_ids, top_k):
        if bm25_exc is not None:
            raise bm25_exc
        return bm25_rows or []

    monkeypatch.setattr(mod.store, "search_chunks", fake_vec)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", fake_bm25)
    return KnowledgeRetriever(None, _FakeEmbedding(), _settings())


# ── _merge_dedup 纯逻辑 ──

def test_merge_rrf_fuses_by_reciprocal_rank():
    """RRF 融合：两路都命中的 chunk 得分最高，单路排名靠前的优于另一路靠后的。"""
    vec = [_row("a", similarity=0.9), _row("b", similarity=0.8)]
    bm25 = [_row("c", score=5.0), _row("a", score=4.0)]
    fused = _merge_dedup(vec, bm25, top_k=3, rrf_k=60)
    # RRF: a = 1/61 + 1/62 ≈ 0.0325, c = 1/61 ≈ 0.0164, b = 1/62 ≈ 0.0161
    assert [r["id"] for r in fused] == ["a", "c", "b"]
    assert fused[0]["sources"] == ["vec", "bm25"]
    assert fused[0]["similarity"] == 0.9
    assert fused[0]["score"] == 4.0  # bm25 原始分补充
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62, rel=1e-6)


def test_merge_disjoint_concatenates():
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("b", score=5.0)]
    fused = _merge_dedup(vec, bm25, top_k=5)
    assert [r["id"] for r in fused] == ["a", "b"]
    assert fused[0]["sources"] == ["vec"]
    assert fused[1]["sources"] == ["bm25"]


def test_merge_single_empty_leg_passthrough():
    vec = [_row("a", similarity=0.9), _row("b", similarity=0.8)]
    fused = _merge_dedup(vec, [], top_k=5)
    assert [r["id"] for r in fused] == ["a", "b"]
    assert all(r["sources"] == ["vec"] for r in fused)


def test_merge_truncates_to_top_k():
    vec = [_row(f"v{i}", similarity=1.0 - i * 0.1) for i in range(5)]
    fused = _merge_dedup(vec, [], top_k=3)
    assert len(fused) == 3


# ── search 行为 ──

async def test_search_bm25_failure_degrades_to_vector_only(monkeypatch, caplog):
    vec = [_row("a", similarity=0.9)]
    r = _retriever(monkeypatch, vec, bm25_exc=RuntimeError("index missing"))
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("q", ["kb-1"])
    assert [x["id"] for x in result] == ["a"]
    assert any("BM25 召回失败" in x.getMessage() for x in caplog.records)


async def test_search_empty_both_legs_returns_empty(monkeypatch):
    """双路皆空返回 [], 供 _route_after_topk 走 no_results 分支。"""
    r = _retriever(monkeypatch, [], [])
    result = await r.search("无关问题", ["kb-1"])
    assert result == []


async def test_search_vector_failure_degrades_to_bm25(monkeypatch, caplog):
    """向量路异常不外抛:降级纯 BM25, 与 BM25 路容错对等。"""
    import rag.document.retriever as mod

    async def boom_vec(pool, embedding, knowledge_base_ids, top_k):
        raise RuntimeError("vector down")

    async def fake_bm25(pool, query_text, knowledge_base_ids, top_k):
        return [_row("b", score=4.2)]

    monkeypatch.setattr(mod.store, "search_chunks", boom_vec)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", fake_bm25)
    r = KnowledgeRetriever(None, _FakeEmbedding(), _settings())
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("q", ["kb-1"])
    assert [x["id"] for x in result] == ["b"]
    assert any("向量召回失败" in x.getMessage() for x in caplog.records)


async def test_search_punctuation_only_query_skips_bm25_leg(monkeypatch, caplog):
    import rag.document.retriever as mod

    calls = {"bm25": 0}

    async def fake_vec(pool, embedding, knowledge_base_ids, top_k):
        return [_row("a", similarity=0.9)]

    async def fake_bm25(pool, query_text, knowledge_base_ids, top_k):
        calls["bm25"] += 1
        return []

    monkeypatch.setattr(mod.store, "search_chunks", fake_vec)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", fake_bm25)
    r = KnowledgeRetriever(None, _FakeEmbedding(), _settings())
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("？！。……", ["kb-1"])
    assert calls["bm25"] == 0
    assert [x["id"] for x in result] == ["a"]


async def test_fused_rows_carry_both_score_keys(monkeypatch):
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("b", score=4.2)]
    r = _retriever(monkeypatch, vec, bm25)
    result = await r.search("孙悟空", ["kb-1"])
    by_id = {x["id"]: x for x in result}
    assert by_id["a"]["score"] is None and by_id["a"]["similarity"] == 0.9
    assert by_id["b"]["similarity"] is None and by_id["b"]["score"] == 4.2


async def test_fused_overlap_preserves_both_raw_scores(monkeypatch):
    """同一 chunk 在两路都命中时，两路原始分均保留不丢失。"""
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("a", score=4.2)]
    r = _retriever(monkeypatch, vec, bm25)
    result = await r.search("孙悟空", ["kb-1"])
    assert len(result) == 1
    assert result[0]["similarity"] == 0.9
    assert result[0]["score"] == 4.2
    assert result[0]["sources"] == ["vec", "bm25"]


# ── fetch_parent_contents ──


# ── 三路 RRF 融合 (graph 路) ──


def test_merge_dedup_three_way_rrf():
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("b", text="B", score=5.0)]
    graph = [_row("a", text="A"), _row("c", text="C", chunk_index=2)]
    fused = _merge_dedup(vec, bm25, top_k=10, rrf_k=60, graph_rows=graph)

    by_id = {r["id"]: r for r in fused}
    assert by_id["a"]["sources"] == ["vec", "graph"]
    assert by_id["c"]["sources"] == ["graph"]
    # a 双路命中,RRF 分 = 1/61 + 1/61,必高于单路的 b/c
    assert fused[0]["id"] == "a"


def test_merge_dedup_two_way_backward_compat():
    vec = [_row("a", similarity=0.9)]
    fused = _merge_dedup(vec, [], top_k=5)
    assert [r["id"] for r in fused] == ["a"]


def test_merge_dedup_graph_only():
    graph = [_row("g", text="G")]
    fused = _merge_dedup([], [], top_k=5, graph_rows=graph)
    assert [r["id"] for r in fused] == ["g"]
    assert fused[0]["sources"] == ["graph"]


class _FakeGraphRetriever:
    """带 calls 记录的假图检索器,照抄本文件 store/embedding monkeypatch 风格。"""

    def __init__(self, rows=None, exc=None):
        self.calls = []
        self._rows = rows or []
        self._exc = exc

    async def search(self, entities, limit):
        self.calls.append((entities, limit))
        if self._exc is not None:
            raise self._exc
        return self._rows


async def test_search_runs_graph_leg_when_entities(monkeypatch):
    vec = [_row("a", similarity=0.9)]
    r = _retriever(monkeypatch, vec, [])
    graph_retriever = _FakeGraphRetriever(rows=[_row("g", text="G")])
    r._graph_retriever = graph_retriever

    result = await r.search("孙悟空", ["kb-1"], entities=["孙悟空"])

    assert len(graph_retriever.calls) == 1
    called_entities, called_limit = graph_retriever.calls[0]
    assert called_entities == ["孙悟空"]
    by_id = {x["id"]: x for x in result}
    assert by_id["g"]["sources"] == ["graph"]


async def test_search_graph_leg_failure_degrades(monkeypatch):
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("b", text="B", score=5.0)]
    r = _retriever(monkeypatch, vec, bm25)
    graph_retriever = _FakeGraphRetriever(exc=RuntimeError("graph down"))
    r._graph_retriever = graph_retriever

    result = await r.search("孙悟空", ["kb-1"], entities=["孙悟空"])

    assert [x["id"] for x in result] == ["a", "b"]


async def test_search_no_entities_skips_graph(monkeypatch):
    vec = [_row("a", similarity=0.9)]
    r = _retriever(monkeypatch, vec, [])
    graph_retriever = _FakeGraphRetriever(rows=[_row("g", text="G")])
    r._graph_retriever = graph_retriever

    result = await r.search("孙悟空", ["kb-1"])

    assert graph_retriever.calls == []
    assert [x["id"] for x in result] == ["a"]


async def test_search_dedupes_duplicate_entities_before_graph_call(monkeypatch):
    """重复实体名在传给 graph_retriever 前去重,避免种子权重被重复计入。"""
    vec = [_row("a", similarity=0.9)]
    r = _retriever(monkeypatch, vec, [])
    graph_retriever = _FakeGraphRetriever(rows=[])
    r._graph_retriever = graph_retriever

    await r.search("孙悟空", ["kb-1"], entities=["孙悟空", "孙悟空"])

    assert graph_retriever.calls[0][0] == ["孙悟空"]


async def test_fetch_parent_contents_delegates_to_store(monkeypatch):
    """按 doc_id 批量补查父文档全文：委托给 store.get_documents_content。"""
    import rag.document.retriever as mod

    called = {}

    async def fake_get_contents(pool, document_ids):
        called["ids"] = list(document_ids)
        return {"doc-1": "全文A"}

    monkeypatch.setattr(mod.store, "get_documents_content", fake_get_contents)
    r = KnowledgeRetriever(None, _FakeEmbedding(), _settings())

    out = await r.fetch_parent_contents(["doc-1", "doc-2"])

    assert out == {"doc-1": "全文A"}
    assert called["ids"] == ["doc-1", "doc-2"]
