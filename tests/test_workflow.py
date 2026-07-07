import rag.agent.workflow as wf
from rag.agent.type import ContextSchema


class _FakeMM:
    async def search(self, session_id, query, top_k=5, filters=None):
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        return [{"text": "S1"}]

    async def persist_turn(self, session_id, query, answer):
        pass


class _FakeLLM:
    async def ainvoke(self, messages):
        return "rw"

    async def ainvoke_structured(self, messages, schema):
        from rag.agent.nodes.query.query import QueryRewriteOutput

        return QueryRewriteOutput(rewrite_query="rw", is_out_of_scope=False)

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


async def test_invoke_passes_config_to_astream(monkeypatch):
    import rag.agent.workflow as wf

    captured = {}

    async def fake_astream(input, *, context, stream_mode, config=None):
        captured["config"] = config
        if False:  # pragma: no cover - 使函数成为异步生成器
            yield

    monkeypatch.setattr(wf.graph, "astream", fake_astream)
    async for _ in wf.invoke("s-1", "q", context=None, config={"callbacks": []}):
        pass
    assert captured["config"] == {"callbacks": []}


async def test_invoke_out_of_scope_skips_recall(monkeypatch):
    """is_out_of_scope=True 时 graph 应跳过 recall，直接走 direct_answer。"""
    import rag.agent.workflow as wf
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _OutOfScopeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(rewrite_query="你好", is_out_of_scope=True)

        async def astream(self, messages):
            yield "你好呀！"

    class _NoCallRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_id, top_k=5):
            self.calls.append(query)
            return [{"text": "SHOULD_NOT_APPEAR"}]

    retriever = _NoCallRetriever()
    ctx = ContextSchema(
        llm=_OutOfScopeLLM(), memory_manager=_FakeMM(), retriever=retriever
    )

    messages: list[str] = []
    statuses: list[str] = []
    async for event in wf.invoke("s1", "你好", ctx):
        if event["type"] == "message":
            messages.append(event["data"])
        elif event["type"] == "status":
            statuses.append(event["data"])

    # recall 完全未被调用
    assert retriever.calls == []
    # 没有"检索知识库中..."的状态
    assert "检索知识库中..." not in statuses
    # 走了 direct_answer 拿到了回复
    assert messages == ["你好呀！"]


async def test_invoke_config_defaults_none(monkeypatch):
    import rag.agent.workflow as wf

    captured = {}

    async def fake_astream(input, *, context, stream_mode, config=None):
        captured["config"] = config
        if False:  # pragma: no cover
            yield

    monkeypatch.setattr(wf.graph, "astream", fake_astream)
    async for _ in wf.invoke("s-1", "q", context=None):
        pass
    assert captured["config"] is None
