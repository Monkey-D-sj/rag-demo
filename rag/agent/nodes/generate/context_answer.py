from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.prompts.generate import context_system_prompt


async def context_answer(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """仅依据会话上下文回答，完全跳过知识库检索。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "根据会话上下文回答中"))

    messages = [
        SystemMessage(
            content=context_system_prompt.format(context=state.get("context", ""))
        ),
        HumanMessage(content=state["raw_query"]),
    ]

    parts: list[str] = []
    async for chunk in runtime.context.llm.astream(messages):
        token = getattr(chunk, "content", chunk)
        if not token:
            continue
        parts.append(token)
        writer(stream_event(StreamEventType.MESSAGE, token))

    state["generated"] = "".join(parts)
    return state
