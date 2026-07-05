import logging

from rag.document.retriever import KnowledgeRetriever


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1, 0.2]]


def _retriever(monkeypatch, rows):
    import rag.document.retriever as mod

    async def fake_search_chunks(pool, embedding, kb_id, top_k):
        return rows

    monkeypatch.setattr(mod.store, "search_chunks", fake_search_chunks)
    return KnowledgeRetriever(None, _FakeEmbedding())


async def test_search_logs_hits_with_structured_fields(monkeypatch, caplog):
    rows = [
        {
            "id": "c1",
            "document_id": "d1",
            "chunk_index": 0,
            "text": "花果山",
            "similarity": 0.87654,
        }
    ]
    r = _retriever(monkeypatch, rows)
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("孙悟空是谁", "kb-1")
    assert result == rows
    rec = next(x for x in caplog.records if "向量召回" in x.getMessage())
    assert rec.levelno == logging.INFO
    assert rec.kb_id == "kb-1"
    assert rec.query == "孙悟空是谁"
    assert rec.top_k == 5
    assert rec.hits == [
        {
            "chunk_id": "c1",
            "chunk_index": 0,
            "similarity": 0.8765,
            "text_preview": "花果山",
        }
    ]
    assert rec.embed_ms >= 0
    assert rec.search_ms >= 0


async def test_search_truncates_text_preview_in_hits(monkeypatch, caplog):
    rows = [
        {
            "id": "c1",
            "document_id": "d1",
            "chunk_index": 0,
            "text": "山" * 300,
            "similarity": 0.5,
        }
    ]
    r = _retriever(monkeypatch, rows)
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        await r.search("q", "kb-1")
    rec = next(x for x in caplog.records if "向量召回" in x.getMessage())
    assert len(rec.hits[0]["text_preview"]) == 80


async def test_search_empty_result_logs_warning(monkeypatch, caplog):
    r = _retriever(monkeypatch, [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("无关问题", "kb-1")
    assert result == []
    rec = next(x for x in caplog.records if "向量召回" in x.getMessage())
    assert rec.levelno == logging.WARNING
    assert rec.hits == []


async def test_search_truncates_long_query_in_log(monkeypatch, caplog):
    r = _retriever(monkeypatch, [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        await r.search("长" * 300, "kb-1")
    rec = next(x for x in caplog.records if "向量召回" in x.getMessage())
    assert len(rec.query) == 200
