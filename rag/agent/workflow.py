from langgraph.graph import END, START, StateGraph

from rag.agent.nodes.add_memory.memory import add_memory
from rag.agent.nodes.generate.generate import direct_answer, generate, no_results
from rag.agent.nodes.query.query import handle_query
from rag.agent.nodes.recall.recall import recall
from rag.agent.nodes.recall_memory.memory import recall_memory
from rag.agent.nodes.rerank.rerank import rerank
from rag.agent.type import ContextSchema, MyState


def _route_after_query(state: MyState) -> str:
    """条件边：范围外直接大模型兜底，范围内走召回。"""
    if state.get("is_out_of_scope"):
        return "direct_answer"
    return "recall"


def _route_after_recall(state: MyState) -> str:
    """条件边：召回为空时返回兜底话术，有结果才走生成。"""
    if state.get("recall_vec_results"):
        return "generate"
    return "no_results"


# 构建状态图
builder = StateGraph(MyState, context_schema=ContextSchema)

# 召回记忆
builder.add_node("recall_memory", recall_memory)
# 查询改写 + 范围判断
builder.add_node("handle_query", handle_query)
# 知识库召回
builder.add_node("recall", recall)
# 语义重排序
builder.add_node("rerank", rerank)
# 基于知识库生成
builder.add_node("generate", generate)
# 范围外直接大模型回答
builder.add_node("direct_answer", direct_answer)
# 记忆持久化（仅 generate 路由到此处，direct_answer / no_results 不写记忆）
builder.add_node("add_memory", add_memory)
# 召回为空时的兜底话术（不调 LLM）
builder.add_node("no_results", no_results)

#                                       ┌─ out-of-scope -> direct_answer ──────────────────────────┐
# START -> recall_memory -> handle_query ┤                                                        END
#                                       └─ in-scope -> recall -> rerank ┬─ generate -> add_memory ─┘
#                                                                       └─ no_results ─────────────┘
builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_conditional_edges(
    "handle_query",
    _route_after_query,
    {"recall": "recall", "direct_answer": "direct_answer"},
)
builder.add_edge("recall", "rerank")
builder.add_conditional_edges(
    "rerank",
    _route_after_recall,
    {"generate": "generate", "no_results": "no_results"},
)
builder.add_edge("generate", "add_memory")
builder.add_edge("add_memory", END)
builder.add_edge("direct_answer", END)
builder.add_edge("no_results", END)

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
        {"session_id": session_id, "raw_query": query, "is_out_of_scope": False},
        context=context,
        stream_mode=["custom"],
        config=config,
    ):
        yield chunk
