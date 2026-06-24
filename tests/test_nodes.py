from types import SimpleNamespace

import rag.nodes.query.query as query_mod
import rag.nodes.recall_memory.memory as recall_mod
from rag.type import ContextSchema


class _FakeLLM:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return "rewritten"


class _FakeMM:
    def __init__(self):
        self.search_calls = []
        self.recent_calls = []

    async def search(self, session_id, query, top_k=5, filters=None):
        self.search_calls.append((session_id, query))
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        self.recent_calls.append(session_id)
        return [{"text": "S1"}]


async def test_recall_memory_awaits_and_composes_context(monkeypatch):
    monkeypatch.setattr(recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    mm = _FakeMM()
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=mm))
    state = {"session_id": "s1", "raw_query": "q1"}

    out = await recall_mod.recall_memory(state, runtime)

    assert mm.search_calls == [("s1", "q1")]
    assert mm.recent_calls == ["s1"]
    assert out["context"] == "S1\nL1"


async def test_handle_query_awaits_ainvoke(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    llm = _FakeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q1", "context": "ctx"}

    out = await query_mod.handle_query(state, runtime)

    assert out["rewrite_query"] == "rewritten"
    assert len(llm.calls) == 1
