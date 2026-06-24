import rag.workflow as wf
from rag.type import ContextSchema


class _FakeMM:
    async def search(self, session_id, query, top_k=5, filters=None):
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        return [{"text": "S1"}]


class _FakeLLM:
    async def ainvoke(self, messages):
        return "rw"


async def test_invoke_runs_full_graph_with_context():
    ctx = ContextSchema(llm=_FakeLLM(), memory_manager=_FakeMM())

    merged = {}
    async for chunk in wf.invoke("s1", "q1", ctx):
        for _node, payload in chunk.items():
            merged.update(payload)

    assert merged["context"] == "S1\nL1"
    assert merged["rewrite_query"] == "rw"
