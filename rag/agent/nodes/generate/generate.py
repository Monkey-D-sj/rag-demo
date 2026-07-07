from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MessageRole, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger

logger = get_logger()

system_prompt = """
你是一个专业的问答助手。你的任务是：基于提供的上下文信息和知识库内容，准确、简洁地回答用户的问题。

## 回答规则

1. **结论先行**：开门见山给出答案，不绕弯子。
2. **引用来源**：在回答中涉及上下文信息的部分，用上标 `[序号]` 标注出处。
3. **末尾附注**：回答结束后，另起一行，按序号列出所有引用的原文出处。
4. **精确回答**：只回答用户问的问题，不要添加上下文之外的无关信息或额外推测。
5. **诚实兜底**：如果上下文信息不足以回答用户问题，请明确告知"根据现有信息无法回答"，不要编造内容。
6. **多源引用**：如一条结论来自多个出处，用 `[1][2]` 并列标注。
7. **知识边界**：回答中若包含自身常识（非上下文内容），需显式说明"据我所知..."，与引用内容区分开。

## 回答格式示例

**用户问题**：刘备的三弟是谁？

**上下文**：
> 刘备和关羽、张飞结拜为兄弟，排行是刘备1，关羽2，张飞3。

**你的回答**：
刘备的三弟是张飞[1][2]。

[1]: 刘备和关羽、张飞结拜为兄弟，排行是刘备1，关羽2，张飞3。
[2]：刘备喝止张飞：三弟，快停手

---

现在，请根据以上规则，基于提供的上下文回答用户的问题。
"""

direct_system_prompt = """
你是一个智能问答助手。用户的问题与知识库内容无关，请直接基于你的知识回答。

## 回答规则

1. **友好自然**：如果是闲聊，友好、自然地回复。
2. **专业准确**：如果是技术或知识类问题，给出专业、准确的回答。
3. **诚实透明**：如果不确定或超出知识范围，诚实告知。
4. **简洁明了**：开门见山，不要绕弯子。

现在，请直接回答用户的问题。
"""


async def generate(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """基于知识库检索结果生成回答：逐 token 流式输出。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "生成回答中"))

    llm = runtime.context.llm
    chunks = state.get("recall_vec_results") or []
    knowledge = "\n".join(c.get("text", "") for c in chunks)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
查询: {state["raw_query"]}
知识库内容: {knowledge}
"""
        ),
    ]

    parts: list[str] = []
    async for chunk in llm.astream(messages):
        token = getattr(chunk, "content", chunk)
        if not token:
            continue
        parts.append(token)
        writer(stream_event(StreamEventType.MESSAGE, token))

    state["generated"] = "".join(parts)

    await _persist_turn(
        runtime.context.memory_manager,
        state["session_id"],
        state["raw_query"],
        state["generated"],
    )
    return state


async def direct_answer(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """知识库范围外直接回答：不依赖检索结果，大模型自身知识兜底。

    范围外问答不写回记忆——闲聊/无关问题对多轮对话没有上下文价值。
    """
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "生成回答中"))

    llm = runtime.context.llm
    messages = [
        SystemMessage(content=direct_system_prompt),
        HumanMessage(content=state["raw_query"]),
    ]

    parts: list[str] = []
    async for chunk in llm.astream(messages):
        token = getattr(chunk, "content", chunk)
        if not token:
            continue
        parts.append(token)
        writer(stream_event(StreamEventType.MESSAGE, token))

    state["generated"] = "".join(parts)
    return state


async def _persist_turn(
    memory_manager, session_id: str, query: str, answer: str
) -> None:
    if memory_manager is None:
        return
    try:
        await memory_manager.add_message(session_id, query, {"role": MessageRole.USER})
        await memory_manager.add_message(session_id, answer, {"role": MessageRole.ASSISTANT})
        await memory_manager.add(session_id, query, {"role": MessageRole.USER})
        await memory_manager.add(session_id, answer, {"role": MessageRole.ASSISTANT})
    except Exception:  # noqa: BLE001 - 记忆写入失败不拖垮回答
        logger.exception("写回会话记忆失败: session=%s", session_id)
