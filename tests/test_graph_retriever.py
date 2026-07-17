from unittest.mock import MagicMock

import pytest

from rag.graph.retriever import GraphRetriever, _score_chunks


class _FakeResult:
    def __init__(self, records):
        self._records = records

    async def data(self):
        return self._records


class _FakeSession:
    def __init__(self, records):
        self._records = records
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def run(self, query, **params):
        self.queries.append((query, params))
        return _FakeResult(self._records)


class _FakeDriver:
    def __init__(self, records):
        self.records = records
        self.sessions = []

    def session(self, database=None):
        s = _FakeSession(self.records)
        self.sessions.append(s)
        return s


def test_score_chunks_seed_outranks_neighbor():
    records = [
        {"seed": "孙悟空", "seed_chunks": ["d1:1", "d1:2"],
         "neighbor_chunk_lists": [["d1:3"], ["d2:0"]]},
    ]
    ranked = _score_chunks(records)
    # 种子实体 chunk(权重2)排在仅邻居命中的 chunk(权重1)之前
    assert ranked.index(("d1", 1)) < ranked.index(("d1", 3))
    assert ranked.index(("d1", 2)) < ranked.index(("d2", 0))


def test_score_chunks_multi_entity_hit_outranks_single():
    records = [
        {"seed": "孙悟空", "seed_chunks": ["d1:1"], "neighbor_chunk_lists": []},
        {"seed": "唐僧", "seed_chunks": ["d1:1", "d1:9"], "neighbor_chunk_lists": []},
    ]
    ranked = _score_chunks(records)
    # d1:1 被两个种子实体命中,应排最前
    assert ranked[0] == ("d1", 1)


def test_score_chunks_skips_malformed_uid():
    records = [
        {"seed": "孙悟空", "seed_chunks": ["badformat", "d1:2"],
         "neighbor_chunk_lists": [None]},
    ]
    ranked = _score_chunks(records)
    assert ranked == [("d1", 2)]


async def test_search_fetches_rows_in_score_order(monkeypatch):
    records = [
        {"seed": "孙悟空", "seed_chunks": ["d1:2"], "neighbor_chunk_lists": [["d2:5"]]},
    ]
    driver = _FakeDriver(records)

    async def _fake_get(pool, uids):
        # 模拟 SQL 乱序返回
        return [
            {"id": "b", "document_id": "d2", "chunk_index": 5, "text": "nb"},
            {"id": "a", "document_id": "d1", "chunk_index": 2, "text": "seed"},
        ]

    monkeypatch.setattr("rag.graph.retriever.store.get_chunks_by_uids", _fake_get)
    gr = GraphRetriever(driver, "neo4j", MagicMock())

    rows = await gr.search(["孙悟空"], limit=10)

    assert [r["id"] for r in rows] == ["a", "b"]  # 按图评分序,非 SQL 返回序


async def test_search_respects_limit(monkeypatch):
    records = [
        {"seed": "孙悟空", "seed_chunks": [f"d1:{i}" for i in range(20)],
         "neighbor_chunk_lists": []},
    ]
    driver = _FakeDriver(records)
    captured = {}

    async def _fake_get(pool, uids):
        captured["uids"] = uids
        return []

    monkeypatch.setattr("rag.graph.retriever.store.get_chunks_by_uids", _fake_get)
    gr = GraphRetriever(driver, "neo4j", MagicMock())

    await gr.search(["孙悟空"], limit=5)

    assert len(captured["uids"]) == 5


async def test_search_empty_entities_returns_empty():
    gr = GraphRetriever(_FakeDriver([]), "neo4j", MagicMock())
    assert await gr.search([], limit=5) == []
