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


def _finish_call(call_id="finish"):
    return _tool_call("finish_evidence_collection", {}, call_id=call_id)


def _ctx(llm, retriever=None):
    return SimpleNamespace(
        context=ContextSchema(llm=llm, memory_manager=None, retriever=retriever)
    )


async def _run(
    monkeypatch,
    llm,
    *,
    max_steps=3,
    timeout=30,
    max_tool_calls=4,
    max_evidence_chars=24_000,
    retriever=None,
):
    emitted: list[dict] = []
    monkeypatch.setattr(agent_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    monkeypatch.setattr(
        agent_mod, "get_settings",
        lambda: SimpleNamespace(
            AGENT_MAX_STEPS=max_steps,
            AGENT_MAX_TOOL_CALLS_PER_STEP=max_tool_calls,
            AGENT_MAX_EVIDENCE_CHARS=max_evidence_chars,
            AGENT_TOTAL_TIMEOUT_SECONDS=timeout,
        ),
    )
    state = {
        "session_id": "s1",
        "raw_query": "那实际门槛呢?",
        "rewrite_query": "《A办法》参照《B条例》执行的实际资质门槛",
    }
    out = await agent_execute(state, _ctx(llm, retriever or _StubRetriever()))
    return out, emitted


async def test_loop_converges_and_collects_rows(monkeypatch):
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "《A办法》资质门槛"}, call_id="c1"),
        _tool_call("fetch_document", {"document_id": "B"}, call_id="c2"),
        _finish_call(),
    ])
    out, emitted = await _run(monkeypatch, llm, retriever=_StubRetriever(contents={"B": "《B条例》第十二条：注册资本 500 万。"}))

    texts = [r["text"] for r in out["recall_vec_results"]]
    assert "门槛 500 万" in texts
    assert any("注册资本 500 万" in t for t in texts)
    assert len(out["recall_vec_results"]) == 2  # 片段行 + 全文行(document_id 相同但 chunk_index 不同)
    assert out["agent_succeeded"] is True
    assert out["agent_skip_cache"] is True
    assert "generated" not in out  # 节点不产出答案
    assert any(ev["data"] == "深度检索中…" for ev in emitted)
    first_human = llm.calls[0][1]
    assert "那实际门槛呢" in first_human.content
    assert "《A办法》参照《B条例》" in first_human.content
    # 全文窗口沿用首次检索到的真实来源名，不把 UUID 当作引用标题。
    assert out["recall_vec_results"][1]["filename"] == "《B条例》"


async def test_loop_dedupes_rows_by_document(monkeypatch):
    rows = [
        {"text": "同一句", "filename": "《X》", "document_id": "X", "metadata": {}, "chunk_index": None},
        {"text": "同一句", "filename": "《X》", "document_id": "X", "metadata": {}, "chunk_index": None},
    ]
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "q"}, call_id="c1"),
        _finish_call(),
    ])
    out, _ = await _run(monkeypatch, llm, retriever=_StubRetriever(rows=rows))
    assert len(out["recall_vec_results"]) == 1


async def test_loop_stops_at_max_steps_without_raise(monkeypatch):
    # 每轮都想调工具,步数上限截断;残缺证据清空并降级主链。
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "q1"}, call_id="c1"),
        _tool_call("retrieve_kb", {"query": "q2"}, call_id="c2"),
    ])
    out, emitted = await _run(monkeypatch, llm, max_steps=2, retriever=_PerQueryRetriever())
    assert len(llm.calls) == 2  # 步数上限拦住第三轮
    assert out["recall_vec_results"] == []
    assert out["agent_succeeded"] is False
    assert out["agent_skip_cache"] is False
    assert any("第 2/2 步" in ev["data"] for ev in emitted)


async def test_loop_llm_error_degrades_to_mainline(monkeypatch):
    class _BoomLLM:
        async def ainvoke_with_tools(self, messages, tools):
            raise RuntimeError("provider down")

    out, _ = await _run(monkeypatch, _BoomLLM())
    assert out["recall_vec_results"] == []
    assert out["agent_succeeded"] is False
    assert out["agent_skip_cache"] is False


async def test_loop_invalid_args_returned_as_observation(monkeypatch):
    # retrieve_kb 缺 query 参数 → handler TypeError → 以 observation 回喂,模型第二轮自纠后收敛
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {}, call_id="c1"),
        _tool_call("retrieve_kb", {"query": "补全后的查询"}, call_id="c2"),
        _finish_call(),
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


async def test_loop_requires_explicit_finish_tool(monkeypatch):
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "q"}),
        AIMessage(content="我觉得证据够了"),
    ])
    out, _ = await _run(monkeypatch, llm)
    assert out["recall_vec_results"] == []
    assert out["agent_succeeded"] is False


async def test_loop_rejects_finish_without_evidence(monkeypatch):
    llm = _ScriptedLLM([_finish_call(), AIMessage(content="stop")])
    out, _ = await _run(monkeypatch, llm)
    assert out["recall_vec_results"] == []
    assert out["agent_succeeded"] is False
    second_call_msgs = llm.calls[1]
    assert any(
        isinstance(m, ToolMessage) and "尚未收集到证据" in m.content
        for m in second_call_msgs
    )


async def test_loop_discards_partial_evidence_when_later_step_fails(monkeypatch):
    class _PartialThenBoom:
        def __init__(self):
            self.calls = 0

        async def ainvoke_with_tools(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return _tool_call("retrieve_kb", {"query": "第一跳"})
            raise RuntimeError("second step failed")

    out, _ = await _run(monkeypatch, _PartialThenBoom())
    assert out["recall_vec_results"] == []
    assert out["agent_succeeded"] is False


async def test_loop_enforces_total_evidence_budget(monkeypatch):
    rows = [
        {"text": "x" * 800, "filename": "A", "document_id": f"d{i}",
         "metadata": {}, "chunk_index": i}
        for i in range(5)
    ]
    llm = _ScriptedLLM([
        _tool_call("retrieve_kb", {"query": "q"}),
        _finish_call(),
    ])
    out, _ = await _run(
        monkeypatch,
        llm,
        retriever=_StubRetriever(rows=rows),
        max_evidence_chars=1_500,
    )
    assert out["agent_succeeded"] is True
    assert sum(len(r["text"]) for r in out["recall_vec_results"]) == 1_500
    assert out["recall_vec_results"][-1]["metadata"]["evidence_truncated"] is True


async def test_loop_limits_tool_calls_per_step(monkeypatch):
    calls = [
        {"name": "retrieve_kb", "args": {"query": f"q{i}"},
         "id": f"c{i}", "type": "tool_call"}
        for i in range(3)
    ]
    llm = _ScriptedLLM([
        AIMessage(content="", tool_calls=calls),
        _finish_call(),
    ])
    out, _ = await _run(
        monkeypatch, llm, max_tool_calls=1, retriever=_PerQueryRetriever()
    )
    assert out["agent_succeeded"] is True
    assert len(out["recall_vec_results"]) == 1
    observations = [m.content for m in llm.calls[1] if isinstance(m, ToolMessage)]
    assert len(observations) == 3
    assert sum("本轮最多执行" in text for text in observations) == 2
