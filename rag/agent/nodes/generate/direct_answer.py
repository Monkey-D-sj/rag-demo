from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.prompts.generate import direct_system_prompt


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
