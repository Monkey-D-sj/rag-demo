from langgraph.graph import END, START, StateGraph

from rag.agent.nodes.generate.generate import generate
from rag.agent.nodes.query.query import handle_query
from rag.agent.nodes.recall.recall import recall
from rag.agent.nodes.recall_memory.memory import recall_memory
from rag.agent.type import MyState, ContextSchema

# 构建状态图
builder = StateGraph(MyState, context_schema=ContextSchema)

builder.add_node("recall_memory", recall_memory)
builder.add_node("handle_query", handle_query)
builder.add_node("recall", recall)
builder.add_node("generate", generate)

# builder.add_edge(START, "recall_memory")
# builder.add_edge("recall_memory", "handle_query")
# builder.add_edge("handle_query", "recall")
builder.add_edge(START, "recall")
builder.add_edge("recall", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


async def invoke(
    session_id: str,
    query: str,
    context: ContextSchema,
    config: dict | None = None,
):
    """归一化事件流:仅保留 custom 通道事件(status/message/error),
    updates 通道(state 增量)不再下发。config 用于透传 LangChain 回调(如 Langfuse)。
    """
    async for mode, chunk in graph.astream(
        {"session_id": session_id, "raw_query": query},
        context=context,
        stream_mode=["custom"],
        config=config,
    ):
        yield chunk
