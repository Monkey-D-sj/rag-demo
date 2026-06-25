import rag.agent.workflow as wf
from rag.agent.type import ContextSchema


class _FakeMM:
    async def search(self, session_id, query, top_k=5, filters=None):
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        return [{"text": "S1"}]


class _FakeLLM:
    async def ainvoke(self, messages):
        return "rw"

    async def astream(self, messages):
        for tok in ("你好", "世界"):
            yield tok


class _FakeRetriever:
    def __init__(self):
        self.calls = []

    async def search(self, query, knowledge_base_id, top_k=5):
        self.calls.append((query, knowledge_base_id))
        return [{"text": "KB1"}]


async def test_invoke_runs_full_graph_with_context():
    retriever = _FakeRetriever()
    ctx = ContextSchema(
        llm=_FakeLLM(), memory_manager=_FakeMM(), retriever=retriever
    )

    merged = {}
    statuses = []
    tokens = []
    async for event in wf.invoke("s1", "q1", ctx):
        if event["type"] == "update":
            merged.update(event["data"])
        elif event["type"] == "status":
            statuses.append(event["data"])
        elif event["type"] == "token":
            tokens.append(event["data"])

    assert merged["context"] == "S1\nL1"
    assert merged["rewrite_query"] == "rw"
    # recall 节点用改写后的查询检索知识库,结果进入 state
    assert retriever.calls == [("rw", "00000000-0000-0000-0000-000000000001")]
    assert merged["recall_vec_results"] == [{"text": "KB1"}]
    # 最终生成逐 token 流式,并累积为完整 generated
    assert tokens == ["你好", "世界"]
    assert merged["generated"] == "你好世界"
    # 节点经由 custom 通道发出的进度事件也应透传
    assert "检索记忆中..." in statuses
    assert "检索知识库中..." in statuses
