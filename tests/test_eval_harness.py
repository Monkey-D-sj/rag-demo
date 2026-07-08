from pathlib import Path

import pytest

from rag.eval.harness import GoldenItem, load_golden, run_eval


def test_load_golden_parses_and_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"id":"q1","query":"金箍棒多重?","gold_snippets":["一万三千五百斤"]}\n'
        "\n"
        '{"query":"谁是大师兄?","gold_snippets":["孙悟空"],"out_of_scope":false}\n',
        encoding="utf-8",
    )
    items = load_golden(p)
    assert len(items) == 2
    assert items[0] == GoldenItem(
        id="q1", query="金箍棒多重?", gold_snippets=["一万三千五百斤"], rewrite_query=None, out_of_scope=False,
    )
    assert items[1].id == "3"  # 无 id 时回退行号
    assert items[1].out_of_scope is False


def test_load_golden_parses_out_of_scope():
    import tempfile, os
    p = Path(tempfile.mktemp(suffix=".jsonl"))
    p.write_text(
        '{"id":"q1","query":"你好啊","gold_snippets":[],"out_of_scope":true}\n',
        encoding="utf-8",
    )
    try:
        items = load_golden(p)
        assert len(items) == 1
        assert items[0].out_of_scope is True
        assert items[0].gold_snippets == []
    finally:
        os.unlink(p)


def test_load_golden_rejects_missing_fields(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"query":"缺片段"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="gold_snippets"):
        load_golden(p)


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1, 0.2]]


class _FakeRetriever:
    def __init__(self, mapping):
        self._mapping = mapping  # query -> list[chunk text]

    async def search(self, query, knowledge_base_id, top_k=5):
        return [{"id": f"c{i}", "chunk_index": i, "text": t}
                for i, t in enumerate(self._mapping.get(query, [])[:top_k])]


class _FakePool:
    pass


async def test_run_eval_returns_four_legs_with_raw(monkeypatch):
    import rag.eval.harness as mod

    embedding = _FakeEmbedding()
    items = [
        GoldenItem("q1", "金箍棒多重?", ["一万三千五百斤"], rewrite_query="如意金箍棒重量"),
        GoldenItem("q2", "大师兄是谁?", ["孙悟空"], rewrite_query="孙悟空 大师兄"),
    ]
    retriever = _FakeRetriever({
        "金箍棒多重?": ["重一万三千五百斤", "无关"],
        "如意金箍棒重量": ["重一万三千五百斤", "金箍棒介绍"],
        "大师兄是谁?": ["无关", "无关"],
        "孙悟空 大师兄": ["孙悟空是大师兄"],
    })

    async def _empty(*a, **kw): return []
    monkeypatch.setattr(mod.store, "search_chunks", _empty)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", _empty)

    out = await run_eval(items, _FakePool(), embedding, retriever, ks=(1,), top_k=5)

    assert "fused" in out and "raw" in out and "vec_only" in out and "bm25_only" in out
    # q1: raw hit, fused hit → hit@1=1.0 for both
    # q2: raw miss, fused hit → hit@1=0.5 for raw, 1.0 for fused (改写提升)
    assert out["raw"]["aggregate"]["hit@1"] == 0.5
    assert out["fused"]["aggregate"]["hit@1"] == 1.0
    assert len(out["fused"]["per_query"]) == 2
