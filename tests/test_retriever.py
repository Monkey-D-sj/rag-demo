import logging

from rag.document.retriever import KnowledgeRetriever, _rrf_fuse


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1, 0.2]]


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

    async def fake_vec(pool, embedding, kb_id, top_k):
        return vec_rows

    async def fake_bm25(pool, query_text, kb_id, top_k):
        if bm25_exc is not None:
            raise bm25_exc
        return bm25_rows or []

    monkeypatch.setattr(mod.store, "search_chunks", fake_vec)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", fake_bm25)
    return KnowledgeRetriever(None, _FakeEmbedding())


# ── _rrf_fuse 纯逻辑 ──

def test_rrf_overlap_ranks_shared_chunk_first():
    vec = [_row("a", similarity=0.9), _row("b", similarity=0.8)]
    bm25 = [_row("c", score=5.0), _row("a", score=4.0)]
    fused = _rrf_fuse(vec, bm25, top_k=3)
    assert [r["id"] for r in fused][0] == "a"  # 双路命中 RRF 最高
    assert fused[0]["sources"] == ["vec", "bm25"]


def test_rrf_disjoint_interleaves_by_rank():
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("b", score=5.0)]
    fused = _rrf_fuse(vec, bm25, top_k=5)
    assert {r["id"] for r in fused} == {"a", "b"}
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]  # 各自 rank 1,并列


def test_rrf_single_empty_leg_passthrough_order():
    vec = [_row("a", similarity=0.9), _row("b", similarity=0.8)]
    fused = _rrf_fuse(vec, [], top_k=5)
    assert [r["id"] for r in fused] == ["a", "b"]
    assert all(r["sources"] == ["vec"] for r in fused)


def test_rrf_truncates_to_top_k():
    vec = [_row(f"v{i}", similarity=1.0 - i * 0.1) for i in range(5)]
    fused = _rrf_fuse(vec, [], top_k=3)
    assert len(fused) == 3


# ── search 行为 ──

async def test_search_fuses_and_logs_both_legs(monkeypatch, caplog):
    vec = [_row("a", text="花果山", similarity=0.87654)]
    bm25 = [_row("b", text="金箍棒", score=4.2)]
    r = _retriever(monkeypatch, vec, bm25)
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("孙悟空的兵器", "kb-1")
    assert {x["id"] for x in result} == {"a", "b"}
    rec = next(x for x in caplog.records if "混合召回" in x.getMessage())
    assert rec.levelno == logging.INFO
    assert rec.kb_id == "kb-1"
    assert rec.vec_hits[0]["chunk_id"] == "a"
    assert rec.vec_hits[0]["similarity"] == 0.8765
    assert rec.bm25_hits[0]["chunk_id"] == "b"
    assert rec.bm25_hits[0]["score"] == 4.2
    assert rec.hits[0]["sources"] in (["vec"], ["bm25"])
    assert rec.hits[0]["text_preview"]
    assert rec.embed_ms >= 0 and rec.search_ms >= 0 and rec.bm25_ms >= 0


async def test_search_bm25_failure_degrades_to_vector_only(monkeypatch, caplog):
    vec = [_row("a", similarity=0.9)]
    r = _retriever(monkeypatch, vec, bm25_exc=RuntimeError("index missing"))
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("q", "kb-1")
    assert [x["id"] for x in result] == ["a"]
    assert any("BM25 召回失败" in x.getMessage() for x in caplog.records)


async def test_search_empty_both_legs_logs_warning(monkeypatch, caplog):
    r = _retriever(monkeypatch, [], [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("无关问题", "kb-1")
    assert result == []
    rec = next(x for x in caplog.records if "混合召回" in x.getMessage())
    assert rec.levelno == logging.WARNING


async def test_search_truncates_long_query_in_log(monkeypatch, caplog):
    r = _retriever(monkeypatch, [], [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        await r.search("长" * 300, "kb-1")
    rec = next(x for x in caplog.records if "混合召回" in x.getMessage())
    assert len(rec.query) == 200


async def test_search_vector_failure_propagates(monkeypatch):
    import pytest

    import rag.document.retriever as mod

    async def boom_vec(pool, embedding, kb_id, top_k):
        raise RuntimeError("vector down")

    async def fake_bm25(pool, query_text, kb_id, top_k):
        return []

    monkeypatch.setattr(mod.store, "search_chunks", boom_vec)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", fake_bm25)
    r = KnowledgeRetriever(None, _FakeEmbedding())
    with pytest.raises(RuntimeError, match="vector down"):
        await r.search("q", "kb-1")


async def test_search_punctuation_only_query_skips_bm25_leg(monkeypatch, caplog):
    import rag.document.retriever as mod

    calls = {"bm25": 0}

    async def fake_vec(pool, embedding, kb_id, top_k):
        return [_row("a", similarity=0.9)]

    async def fake_bm25(pool, query_text, kb_id, top_k):
        calls["bm25"] += 1
        return []

    monkeypatch.setattr(mod.store, "search_chunks", fake_vec)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", fake_bm25)
    r = KnowledgeRetriever(None, _FakeEmbedding())
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("？！。……", "kb-1")
    assert calls["bm25"] == 0
    assert [x["id"] for x in result] == ["a"]


async def test_fused_rows_carry_both_score_keys(monkeypatch):
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("b", score=4.2)]
    r = _retriever(monkeypatch, vec, bm25)
    result = await r.search("孙悟空", "kb-1")
    by_id = {x["id"]: x for x in result}
    assert by_id["a"]["score"] is None and by_id["a"]["similarity"] == 0.9
    assert by_id["b"]["similarity"] is None and by_id["b"]["score"] == 4.2


async def test_fused_overlap_preserves_both_raw_scores(monkeypatch):
    """同一 chunk 在两路都命中时，两路原始分均保留不丢失。"""
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("a", score=4.2)]
    r = _retriever(monkeypatch, vec, bm25)
    result = await r.search("孙悟空", "kb-1")
    assert len(result) == 1
    assert result[0]["similarity"] == 0.9
    assert result[0]["score"] == 4.2  # 之前会丢失 BM25 分
    assert result[0]["sources"] == ["vec", "bm25"]


async def test_rank_summary_preview_truncated_to_80(monkeypatch, caplog):
    vec = [_row("a", text="山" * 300, similarity=0.9)]
    r = _retriever(monkeypatch, vec, [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        await r.search("q", "kb-1")
    rec = next(x for x in caplog.records if "混合召回" in x.getMessage())
    assert len(rec.vec_hits[0]["text_preview"]) == 80
    assert len(rec.hits[0]["text_preview"]) == 80
