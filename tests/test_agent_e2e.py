"""agent 分支端到端冒烟(integration)。

需要 docker compose 基建(pg + embedding)+ 已 seed 的 EVAL KB。
通过包装真实 LLM、只固定 handle_query 的结构化判定，强制真实图走
agent_execute(真实原生 function calling + 真实工具),验证:
- 全程不抛错、无 error 事件;
- 最终 generated 非空、citations index 从 1 连续、cache_hit=False;
- agent 分支确实进入并成功收敛，证据和引用均非空。
"""

import asyncio
from types import SimpleNamespace

import pytest

import rag.agent.workflow as wf_mod


def _force_agent_enabled(monkeypatch):
    """路由读取的 get_settings 只暴露开关;节点自身的 get_settings 不受影响。"""
    monkeypatch.setattr(
        wf_mod, "get_settings",
        lambda: SimpleNamespace(AGENT_MODE_ENABLED=True, QUERY_DECOMPOSITION_ENABLED=False),
    )


@pytest.mark.integration
async def test_agent_branch_real_graph_smoke(monkeypatch):
    from rag.agent.nodes.query.query import QueryRewriteOutput
    from rag.agent.workflow import build_initial_state
    from rag.config import get_settings
    from rag.eval.answer_harness import build_real_collection_context

    _force_agent_enabled(monkeypatch)
    graph_obj, context, pool = await build_real_collection_context(get_settings())
    try:
        real_llm = context.llm

        class _ForceAgentRoute:
            async def ainvoke_structured(self, messages, schema):
                return QueryRewriteOutput(
                    rewrite_query="花果山位于东胜神洲傲来国附近",
                    is_out_of_scope=False,
                    needs_agent=True,
                )

            async def ainvoke_with_tools(self, messages, tools):
                return await real_llm.ainvoke_with_tools(messages, tools)

            def astream(self, messages):
                return real_llm.astream(messages)

        context.llm = _ForceAgentRoute()
        initial = build_initial_state(
            "eval-agent-smoke", "西游记里提到的花果山在什么位置？"
        )

        messages: list[str] = []
        statuses: list[str] = []
        last_state: dict = {}
        error: str | None = None
        async for mode, payload in graph_obj.astream(
            initial, context=context, stream_mode=["custom", "values"]
        ):
            if mode == "values":
                if isinstance(payload, dict):
                    last_state = payload
                continue
            event = payload if isinstance(payload, dict) else {}
            et, data = event.get("type"), event.get("data")
            if et == "message" and data:
                messages.append(str(data))
            elif et == "status" and data:
                statuses.append(str(data))
            elif et == "error":
                error = str(data)

        answer = str(last_state.get("generated") or "")
        citations = last_state.get("citations") or []
        assert error is None
        assert answer or "".join(messages)
        assert not last_state.get("cache_hit", False)
        assert "深度检索中…" in statuses
        assert last_state.get("agent_succeeded") is True
        assert last_state.get("recall_vec_results")
        assert answer
        assert citations
        indices = [c.get("index") for c in citations]
        assert indices == list(range(1, len(indices) + 1))
    finally:
        await pool.close()
