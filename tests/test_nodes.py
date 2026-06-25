from types import SimpleNamespace

import rag.nodes.generate.generate as generate_mod
import rag.nodes.query.query as query_mod
import rag.nodes.recall.recall as kb_recall_mod
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


async def test_recall_searches_kb_with_rewrite_query(monkeypatch):
    monkeypatch.setattr(
        kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _FakeRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_id, top_k=5):
            self.calls.append((query, knowledge_base_id))
            return [{"text": "KB1"}]

    retriever = _FakeRetriever()
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, retriever=retriever)
    )
    state = {"session_id": "s1", "raw_query": "q1", "rewrite_query": "rw"}

    out = await kb_recall_mod.recall(state, runtime)

    assert out["recall_vec_results"] == [{"text": "KB1"}]
    assert retriever.calls[0][0] == "rw"


async def test_recall_no_retriever_yields_empty(monkeypatch):
    monkeypatch.setattr(
        kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, retriever=None)
    )
    state = {"session_id": "s1", "raw_query": "q1", "rewrite_query": "rw"}

    out = await kb_recall_mod.recall(state, runtime)

    assert out["recall_vec_results"] == []


async def test_generate_streams_tokens_and_accumulates(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        generate_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev))
    )

    class _StreamLLM:
        async def astream(self, messages):
            for tok in ("答", "案"):
                yield tok

    runtime = SimpleNamespace(
        context=ContextSchema(llm=_StreamLLM(), memory_manager=None)
    )
    state = {"session_id": "s1", "rewrite_query": "q", "context": "ctx"}

    out = await generate_mod.generate(state, runtime)

    assert out["generated"] == "答案"
    assert {"type": "token", "data": "答"} in emitted
    assert {"type": "token", "data": "案"} in emitted
