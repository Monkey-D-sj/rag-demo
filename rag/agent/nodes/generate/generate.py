from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.prompts.generate import direct_system_prompt, system_prompt


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


_NO_RESULT_MSG = """
未在当前知识库中找到能够直接回答您问题的依据。
建议您补充更多信息后重新提问，我将继续为您检索相关内容。
"""


async def no_results(state: MyState) -> MyState:
    """知识库无召回结果时直接返回兜底话术，不调 LLM、不写记忆。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "未找到相关内容"))
    writer(stream_event(StreamEventType.MESSAGE, _NO_RESULT_MSG))
    state["generated"] = _NO_RESULT_MSG
    return state
