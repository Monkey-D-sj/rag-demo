"""agent 分支端到端冒烟(integration)。

需要 docker compose 基建(pg + embedding)+ 已 seed 的 EVAL KB。
通过把初始 state 的 needs_agent 置 True 并开启 AGENT_MODE_ENABLED 路由,强制真实图走
agent_execute(原生 function calling + 真实工具),验证:
- 全程不抛错、无 error 事件;
- 最终 generated 非空、citations index 从 1 连续、cache_hit=False;
- agent 分支确实进入("深度检索中…" status)。
无论模型能否真的收敛到上下文(空结果会降级回主链),完成不变量都必须成立。
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
    from rag.agent.workflow import build_initial_state
    from rag.config import get_settings
    from rag.eval.answer_harness import build_real_collection_context

    _force_agent_enabled(monkeypatch)
    graph_obj, context, pool = await build_real_collection_context(get_settings())
    try:
        initial = build_initial_state("eval-agent-smoke", "《A办法》要求参照执行,实际门槛按哪个文件?")
        initial["needs_agent"] = True  # 强制进入 agent 分支,不依赖 handle_query 判定

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
        assert answer or "".join(messages)  # 有产出(agent 直收或降级主链)
        assert not last_state.get("cache_hit", False)
        # agent 分支确实被进入
        assert "深度检索中…" in statuses
        # 若 agent 攒到上下文,则走 generate 直接产出且引用连续
        if last_state.get("recall_vec_results"):
            assert answer
            indices = [c.get("index") for c in citations]
            if indices:
                assert indices == list(range(1, len(indices) + 1))
    finally:
        await pool.close()
