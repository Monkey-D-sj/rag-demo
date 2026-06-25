from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState

system_prompt = """
你是一个专业的关务助手。请基于上下文与改写后的查询,简洁回答用户问题。
"""


async def generate(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """最终生成:逐 token 流式输出,同时累积为完整 generated。"""
    writer = get_stream_writer()
    writer({"type": "status", "data": "生成回答中"})

    llm = runtime.context.llm
    chunks = state.get("recall_vec_results") or []
    knowledge = "\n".join(c.get("text", "") for c in chunks)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
查询: {state["rewrite_query"]}
记忆上下文: {state["context"]}
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
        writer({"type": "token", "data": token})

    state["generated"] = "".join(parts)
    return state
