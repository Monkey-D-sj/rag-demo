from langgraph.graph import END, START, StateGraph

from rag.nodes.generate.generate import generate
from rag.nodes.query.query import handle_query
from rag.nodes.recall.recall import recall
from rag.nodes.recall_memory.memory import recall_memory
from rag.type import MyState, ContextSchema

# 构建状态图
builder = StateGraph(MyState, context_schema=ContextSchema)

builder.add_node("recall_memory", recall_memory)
builder.add_node("handle_query", handle_query)
builder.add_node("recall", recall)
builder.add_node("generate", generate)

builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_edge("handle_query", "recall")
builder.add_edge("recall", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


async def invoke(session_id: str, query: str, context: ContextSchema):
    """归一化事件流:

    - custom 通道:节点 writer 发出的 {"type": "status"|"token", ...},原样透传;
      最终生成节点接入时,逐 token 走 {"type": "token"} 即可。
    - updates 通道:节点产出的 state 增量,包装为 {"type": "update", "node", "data"}。
    """
    async for mode, chunk in graph.astream(
        {"session_id": session_id, "raw_query": query},
        context=context,
        stream_mode=["custom", "updates"],
    ):
        if mode == "custom":
            yield chunk
        else:
            for node, payload in chunk.items():
                yield {"type": "update", "node": node, "data": payload}
