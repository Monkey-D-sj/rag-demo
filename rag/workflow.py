from langgraph.graph import END, START, StateGraph

from rag.nodes.query.query import handle_query
from rag.nodes.recall.memory import recall_memory
from rag.type import MyState

# 构建状态图
builder = StateGraph(MyState)

# 注册节点
builder.add_node("handle_query", handle_query)
builder.add_node("recall_memory", recall_memory)

# 连线: START → recall_memory → handle_query → END
builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_edge("handle_query", END)

# 编译
app = builder.compile()
