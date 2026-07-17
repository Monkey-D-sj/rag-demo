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


def test_load_golden_parses_out_of_scope(tmp_path: Path):
    p = tmp_path / "oos.jsonl"
    p.write_text(
        '{"id":"q1","query":"你好啊","gold_snippets":[],"out_of_scope":true}\n',
        encoding="utf-8",
    )
    items = load_golden(p)
    assert len(items) == 1
    assert items[0].out_of_scope is True
    assert items[0].gold_snippets == []


def test_load_golden_rejects_missing_fields(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"query":"缺片段"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="gold_snippets"):
        load_golden(p)


class _FakeEmbedding:
    @property
    def model(self): return "fake"

    @property
    def dim(self): return 2

    @property
    def batch_size(self): return 10

    async def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _FakeRetriever:
    has_graph = False  # 无图召回,graph_fused 轨不参与

    def __init__(self, mapping):
        self._mapping = mapping  # query -> list[chunk text]

    async def search(self, query, knowledge_base_ids=None, top_k=5, query_emb=None):
        return [{"id": f"c{i}", "chunk_index": i, "text": t}
                for i, t in enumerate(self._mapping.get(query, [])[:top_k])]


class _FakePool:
    pass


async def test_run_eval_returns_six_legs_with_raw(monkeypatch):
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
    # 隔离 embedding 缓存:不然 fake 向量会写进 git 跟踪的 query_embeddings.json
    monkeypatch.setattr(mod, "load_cache", lambda: {})
    monkeypatch.setattr(mod, "save_cache", lambda cache: None)

    out = await run_eval(items, _FakePool(), embedding, retriever, ks=(1,), top_k=5)

    assert "fused" in out and "raw" in out
    assert "vec_only" in out and "raw_vec" in out
    assert "bm25_only" in out and "raw_bm25" in out
    # q1: raw hit, fused hit → hit@1=1.0 for both
    # q2: raw miss, fused hit → hit@1=0.5 for raw, 1.0 for fused
    assert out["raw"]["aggregate"]["hit@1"] == 0.5
    assert out["fused"]["aggregate"]["hit@1"] == 1.0
    assert len(out["fused"]["per_query"]) == 2


def test_golden_item_entities_optional(tmp_path):
    from rag.eval.harness import load_golden

    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"id": "1", "query": "q1", "gold_snippets": ["s"], "entities": ["孙悟空"]}\n'
        '{"id": "2", "query": "q2", "gold_snippets": ["s"]}\n',
        encoding="utf-8",
    )
    items = load_golden(p)
    assert items[0].entities == ["孙悟空"]
    assert items[1].entities == []


async def test_run_eval_graph_leg_gated_by_has_graph(monkeypatch):
    import rag.eval.harness as mod

    embedding = _FakeEmbedding()
    items = [
        GoldenItem(
            "q1", "大师兄是谁?", ["孙悟空是大师兄"],
            rewrite_query="孙悟空 大师兄", entities=["孙悟空"],
        ),
        GoldenItem("q2", "金箍棒多重?", ["一万三千五百斤"], rewrite_query="如意金箍棒重量"),
    ]
    mapping = {
        "孙悟空 大师兄": ["孙悟空是大师兄"],
        "如意金箍棒重量": ["重一万三千五百斤"],
    }

    class _FakeGraphRetriever:
        def __init__(self, mapping, has_graph):
            self._mapping = mapping
            self.has_graph = has_graph
            self.entities_calls: list[tuple[str, list[str]]] = []

        async def search(self, query, knowledge_base_ids=None, top_k=5,
                          query_emb=None, entities=None):
            if entities is not None:
                self.entities_calls.append((query, entities))
            return [{"id": f"c{i}", "chunk_index": i, "text": t}
                    for i, t in enumerate(self._mapping.get(query, [])[:top_k])]

    async def _empty(*a, **kw): return []
    monkeypatch.setattr(mod.store, "search_chunks", _empty)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", _empty)
    monkeypatch.setattr(mod, "load_cache", lambda: {})
    monkeypatch.setattr(mod, "save_cache", lambda cache: None)

    # 1) has_graph=False → result 不含 graph_fused
    retriever_no_graph = _FakeGraphRetriever(mapping, has_graph=False)
    out_no_graph = await run_eval(items, _FakePool(), embedding, retriever_no_graph, ks=(1,), top_k=5)
    assert "graph_fused" not in out_no_graph
    assert retriever_no_graph.entities_calls == []

    # 2) has_graph=True 且条目带 entities → result 含 graph_fused,且该轨调用 search 时传了 entities
    retriever_with_graph = _FakeGraphRetriever(mapping, has_graph=True)
    out_with_graph = await run_eval(items, _FakePool(), embedding, retriever_with_graph, ks=(1,), top_k=5)
    assert "graph_fused" in out_with_graph
    # 只有 q1 带 entities,q2 无 entities 不进 graph 轨
    assert len(out_with_graph["graph_fused"]["per_query"]) == 1
    assert out_with_graph["graph_fused"]["per_query"][0]["id"] == "q1"
    assert retriever_with_graph.entities_calls == [("孙悟空 大师兄", ["孙悟空"])]
