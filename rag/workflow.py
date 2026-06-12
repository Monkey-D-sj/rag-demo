from langgraph.graph import END, START, StateGraph

from rag.memory.manager import get_memory_manager
from rag.models.normal import NormalModel
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

# ── 运行时依赖 ──
memory_manager = get_memory_manager()
llm = NormalModel()
context: ContextSchema = ContextSchema(
    memory_manager=memory_manager,
    llm=llm
)


def invoke(session_id: str, query: str):
    for chunk in graph.stream(
        {"session_id": session_id, "raw_query": query},
        context=context,
    ):
        yield chunk
