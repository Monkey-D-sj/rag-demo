from langgraph.graph import END, START, StateGraph

from rag.nodes.query.query import handle_query
from rag.nodes.recall_memory.memory import recall_memory
from rag.type import MyState, ContextSchema

# 构建状态图
builder = StateGraph(MyState, context_schema=ContextSchema)

builder.add_node("handle_query", handle_query)
builder.add_node("recall_memory", recall_memory)

builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_edge("handle_query", END)

graph = builder.compile()


async def invoke(session_id: str, query: str, context: ContextSchema):
    async for chunk in graph.astream(
        {"session_id": session_id, "raw_query": query},
        context=context,
    ):
        yield chunk
