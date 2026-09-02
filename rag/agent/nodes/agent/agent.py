"""agent 执行节点：有界原生 function calling 工具循环。

只负责"攒对上下文"：把各轮工具结果去重写入 recall_vec_results，交给下游 generate
统一产出带引用的最终答案。节点本身不写 state["generated"]。
失败/超步/空结果一律不向用户抛错——recall_vec_results 可能为空，由路由降级到主链。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.nodes.agent.tools import HANDLERS, _render, build_tools
from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.config import get_settings
from rag.prompts.agent import system_prompt

logger = get_logger()


def _dedupe(rows: list[dict]) -> list[dict]:
    """按 (document_id, chunk_index) 去重;无 document_id 时退化为按 text 去重。"""
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(r.get("document_id") or "")
        key = (doc_id, r.get("chunk_index")) if doc_id else ("", text)
        if key in seen:
            continue
        seen.add(key)
        row = dict(r)
        row.setdefault("filename", "知识库文档")
        row.setdefault("metadata", {})
        row["text"] = text
        out.append(row)
    return out


async def agent_execute(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """多步工具检索，结果去重写入 recall_vec_results，交给 generate。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "深度检索中…"))

    settings = get_settings()
    llm = runtime.context.llm
    session_id = state.get("session_id", "")
    max_steps = settings.AGENT_MAX_STEPS
    timeout_seconds = settings.AGENT_STEP_TIMEOUT_SECONDS

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.get("raw_query", "")),
    ]
    tools = build_tools(runtime.context, session_id)
    collected: list[dict] = []

    try:
        async with asyncio.timeout(timeout_seconds):
            for step in range(1, max_steps + 1):
                writer(stream_event(StreamEventType.STATUS, f"第 {step}/{max_steps} 步"))
                rsp = await llm.ainvoke_with_tools(messages, tools)
                calls = getattr(rsp, "tool_calls", None) or []
                if not calls:
                    break  # 模型判定证据已够，收敛
                messages.append(AIMessage(content=rsp.content, tool_calls=rsp.tool_calls))
                for i, c in enumerate(calls, 1):
                    name = c.get("name")
                    args = c.get("args") or {}
                    call_id = c.get("id") or f"call_{step}_{i}"
                    handler = HANDLERS.get(name)
                    if handler is None:
                        rows: list[dict] = []
                        obs = f"工具不存在: {name}"
                    else:
                        try:
                            rows = await handler(runtime.context, session_id, **args)
                        except Exception as exc:  # noqa: BLE001 - 参数/执行失败作为 observation 回喂,让模型自纠
                            rows = []
                            obs = f"工具 {name} 调用失败: {type(exc).__name__}: {exc}"
                        else:
                            collected.extend(rows)
                            obs = _render(rows)
                    messages.append(ToolMessage(content=obs, tool_call_id=call_id))
                if step == max_steps:
                    logger.warning(
                        "agent 达到步数上限，使用已收集上下文", extra={"query": state.get("raw_query", "")[:200]}
                    )
    except Exception:  # noqa: BLE001 - 超时/异常一律降级,不向用户抛错
        logger.warning("agent 循环失败，降级使用已收集上下文", exc_info=True)

    state["recall_vec_results"] = _dedupe(collected)
    state["agent_skip_cache"] = True
    return state
