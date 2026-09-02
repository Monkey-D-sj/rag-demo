"""agent 执行节点：有界原生 function calling 工具循环。

只负责"攒对上下文"：把各轮工具结果去重写入 recall_vec_results，交给下游 generate
统一产出带引用的最终答案。节点本身不写 state["generated"]。
只有拿到证据且模型明确停止调用工具才算成功；失败/超步/空结果清空临时证据，
由路由降级到主链，避免使用残缺证据作答。
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
    """按文档与位置去重；分页全文窗口使用 offset 区分。"""
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(r.get("document_id") or "")
        meta = r.get("metadata") or {}
        chunk_index = r.get("chunk_index")
        if doc_id and chunk_index is None and "document_offset" in meta:
            key = (doc_id, "offset", meta["document_offset"])
        else:
            key = (doc_id, chunk_index) if doc_id else ("", text)
        if key in seen:
            continue
        seen.add(key)
        row = dict(r)
        row.setdefault("filename", "知识库文档")
        row.setdefault("metadata", {})
        row["text"] = text
        out.append(row)
    return out


def _limit_evidence(rows: list[dict], max_chars: int) -> list[dict]:
    """按总字符预算保留证据，最后一条必要时截断。"""
    out: list[dict] = []
    remaining = max_chars
    for original in _dedupe(rows):
        if remaining <= 0:
            break
        row = dict(original)
        text = row["text"]
        if len(text) > remaining:
            row["text"] = text[:remaining]
            meta = dict(row.get("metadata") or {})
            meta["evidence_truncated"] = True
            row["metadata"] = meta
        out.append(row)
        remaining -= len(row["text"])
    return out


async def agent_execute(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """多步工具检索，结果去重写入 recall_vec_results，交给 generate。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "深度检索中…"))

    settings = get_settings()
    llm = runtime.context.llm
    session_id = state.get("session_id", "")
    max_steps = settings.AGENT_MAX_STEPS
    max_tool_calls = getattr(settings, "AGENT_MAX_TOOL_CALLS_PER_STEP", 4)
    max_evidence_chars = getattr(settings, "AGENT_MAX_EVIDENCE_CHARS", 24_000)
    timeout_seconds = settings.AGENT_TOTAL_TIMEOUT_SECONDS  # 包住整个循环:总时长预算,非每步

    raw_query = state.get("raw_query", "")
    rewritten_query = state.get("rewrite_query") or raw_query
    query_message = f"原始问题：{raw_query}"
    if rewritten_query != raw_query:
        query_message += f"\n已消解指代并适合检索的问题：{rewritten_query}"
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query_message),
    ]
    tools = build_tools(runtime.context, session_id)
    collected: list[dict] = []
    known_sources: dict[str, str] = {}
    succeeded = False

    try:
        async with asyncio.timeout(timeout_seconds):
            for step in range(1, max_steps + 1):
                writer(stream_event(StreamEventType.STATUS, f"第 {step}/{max_steps} 步"))
                rsp = await llm.ainvoke_with_tools(messages, tools)
                calls = getattr(rsp, "tool_calls", None) or []
                if not calls:
                    break
                messages.append(AIMessage(content=rsp.content, tool_calls=rsp.tool_calls))
                finish_requested = False
                for i, c in enumerate(calls, 1):
                    name = c.get("name")
                    args = c.get("args") or {}
                    call_id = c.get("id") or f"call_{step}_{i}"
                    handler = HANDLERS.get(name)
                    if i > max_tool_calls:
                        rows = []
                        obs = f"本轮最多执行 {max_tool_calls} 个工具调用，本调用已跳过。"
                    elif handler is None:
                        rows: list[dict] = []
                        obs = f"工具不存在: {name}"
                    else:
                        try:
                            rows = await handler(runtime.context, session_id, **args)
                        except Exception as exc:  # noqa: BLE001 - 参数/执行失败作为 observation 回喂,让模型自纠
                            rows = []
                            obs = f"工具 {name} 调用失败: {type(exc).__name__}: {exc}"
                        else:
                            if name == "finish_evidence_collection":
                                if collected:
                                    finish_requested = True
                                    obs = "证据已确认齐备，结束检索。"
                                else:
                                    obs = "尚未收集到证据，不能结束检索。"
                            elif name == "fetch_document":
                                for row in rows:
                                    doc_id = str(row.get("document_id") or "")
                                    source = known_sources.get(doc_id)
                                    if source:
                                        row["filename"] = source
                            else:
                                for row in rows:
                                    doc_id = str(row.get("document_id") or "")
                                    if doc_id:
                                        known_sources.setdefault(
                                            doc_id,
                                            row.get("filename") or "知识库文档",
                                        )
                            if name != "finish_evidence_collection":
                                collected = _limit_evidence(
                                    [*collected, *rows], max_evidence_chars
                                )
                                obs = _render(rows)
                    messages.append(ToolMessage(content=obs, tool_call_id=call_id))
                if finish_requested:
                    succeeded = True
                    break
                if step == max_steps:
                    logger.warning(
                        "agent 达到步数上限，降级主链",
                        extra={"query": raw_query[:200]},
                    )
    except Exception:  # noqa: BLE001 - 超时/异常一律降级,不向用户抛错
        logger.warning("agent 循环失败，降级主链", exc_info=True)

    state["agent_succeeded"] = succeeded
    state["recall_vec_results"] = collected if succeeded else []
    state["agent_skip_cache"] = succeeded
    return state
