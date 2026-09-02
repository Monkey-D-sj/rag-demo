"""agent_execute 有界工具循环测试：收敛、步数上限、异常降级、非法参数自纠。"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

import rag.agent.nodes.agent.agent as agent_mod
from rag.agent.nodes.agent.agent import agent_execute
from rag.agent.type import ContextSchema


class _StubRetriever:
    def __init__(self, rows=None, contents=None):
        self.rows = rows or [
            {"text": "门槛 500 万", "filename": "《B条例》", "document_id": "B",
             "metadata": {"article": "第十二条"}, "chunk_index": 1},
        ]
        self.contents = contents or {"B": "《B条例》第十二条：注册资本 500 万。"}

    async def search(self, query, knowledge_base_ids=None, top_k=5):
        return self.rows

    async def fetch_parent_contents(self, document_ids):
        return {d: self.contents.get(d, "") for d in document_ids}


class _PerQueryRetriever:
    """每轮检索返回不同文档,避免去重把多轮结果合并。"""

    async def search(self, query, knowledge_base_ids=None, top_k=5):
        return [{
            "text": f"内容 {query}", "filename": f"《{query}》",
            "document_id": query, "metadata": {}, "chunk_index": 1,
        }]

    async def fetch_parent_contents(self, document_ids):
        return {}


class _ScriptedLLM:
    """按序返回预设 AIMessage;每步记录传入的 messages 供断言 observation。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[list] = []

    async def ainvoke_with_tools(self, messages, tools):
        self.calls.append(list(messages))
        return self._script.pop(0) if self._script else AIMessage(content="证据已齐备")


def _tool_call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _ctx(llm, retriever=None):
    return SimpleNamespace(
        context=ContextSchema(llm=llm, memory_manager=None, retriever=retriever)
    )


async def _run(monkeypatch, llm, *, max_steps=3, timeout=30, retriever=None):
    emitted: list[dict] = []
    monkeypatch.setattr(agent_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    monkeypatch.setattr(
        agent_mod, "get_settings",
        lambda: SimpleNamespace(AGENT_MAX_STEPS=max_steps, AGENT_TOTAL_TIMEOUT_SECONDS=timeout),
    )
    state = {"session_id": "s1", "raw_query": "门槛按哪个执行?"}
    out = await agent_execute(state, _ctx(llm, retriever or _StubRetriever()))
    return out, emitted


async def test_loop_converges_and_collects_rows(monkeypatch):
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "《A办法》资质门槛"}, call_id="c1"),
        _tool_call("fetch_document", {"document_id": "B"}, call_id="c2"),
        AIMessage(content="证据已齐备，可以作答"),
    ])
    out, emitted = await _run(monkeypatch, llm, retriever=_StubRetriever(contents={"B": "《B条例》第十二条：注册资本 500 万。"}))

    texts = [r["text"] for r in out["recall_vec_results"]]
    assert "门槛 500 万" in texts
    assert any("注册资本 500 万" in t for t in texts)
    assert len(out["recall_vec_results"]) == 2  # 片段行 + 全文行(document_id 相同但 chunk_index 不同)
    assert out["agent_skip_cache"] is True
    assert "generated" not in out  # 节点不产出答案
    assert any(ev["data"] == "深度检索中…" for ev in emitted)


async def test_loop_dedupes_rows_by_document(monkeypatch):
    rows = [
        {"text": "同一句", "filename": "《X》", "document_id": "X", "metadata": {}, "chunk_index": None},
        {"text": "同一句", "filename": "《X》", "document_id": "X", "metadata": {}, "chunk_index": None},
    ]
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "q"}, call_id="c1"),
        AIMessage(content="done"),
    ])
    out, _ = await _run(monkeypatch, llm, retriever=_StubRetriever(rows=rows))
    assert len(out["recall_vec_results"]) == 1


async def test_loop_stops_at_max_steps_without_raise(monkeypatch):
    # 每轮都想调工具,步数上限截断;不抛错,用已收集上下文
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "q1"}, call_id="c1"),
        _tool_call("retrieve_kb", {"query": "q2"}, call_id="c2"),
    ])
    out, emitted = await _run(monkeypatch, llm, max_steps=2, retriever=_PerQueryRetriever())
    assert len(llm.calls) == 2  # 步数上限拦住第三轮
    assert len(out["recall_vec_results"]) == 2
    assert any("第 2/2 步" in ev["data"] for ev in emitted)


async def test_loop_llm_error_degrades_to_collected(monkeypatch):
    class _BoomLLM:
        async def ainvoke_with_tools(self, messages, tools):
            raise RuntimeError("provider down")

    out, _ = await _run(monkeypatch, _BoomLLM())
    assert out["recall_vec_results"] == []
    assert out["agent_skip_cache"] is True


async def test_loop_invalid_args_returned_as_observation(monkeypatch):
    # retrieve_kb 缺 query 参数 → handler TypeError → 以 observation 回喂,模型第二轮自纠后收敛
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {}, call_id="c1"),
        _tool_call("retrieve_kb", {"query": "补全后的查询"}, call_id="c2"),
        AIMessage(content="done"),
    ])
    out, _ = await _run(monkeypatch, llm)
    assert out["recall_vec_results"]  # 第二轮成功取到行
    second_call_msgs = llm.calls[1]
    assert any(isinstance(m, ToolMessage) and "调用失败" in m.content for m in second_call_msgs)


async def test_loop_unknown_tool_observation(monkeypatch):
    llm = _ScriptedLLM([
        _tool_call("no_such_tool", {}, call_id="c1"),
        AIMessage(content="done"),
    ])
    out, _ = await _run(monkeypatch, llm)
    assert out["recall_vec_results"] == []
    last_msgs = llm.calls[1]
    assert any(isinstance(m, ToolMessage) and "工具不存在" in m.content for m in last_msgs)
