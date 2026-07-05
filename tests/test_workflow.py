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

    statuses: list[str] = []
    messages: list[str] = []
    async for event in wf.invoke("s1", "q1", ctx):
        if event["type"] == "status":
            statuses.append(event["data"])
        elif event["type"] == "message":
            messages.append(event["data"])

    # recall 节点用改写后的查询检索知识库
    assert retriever.calls == [("rw", "00000000-0000-0000-0000-000000000001")]

    # 节点经由 custom 通道发出的进度事件
    assert "检索记忆中..." in statuses
    assert "检索知识库中..." in statuses
    assert "生成回答中" in statuses

    # 最终生成逐 token 流式
    assert messages == ["你好", "世界"]
