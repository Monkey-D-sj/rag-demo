from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from rag.agent.nodes.add_memory.memory import add_memory
from rag.agent.nodes.cache_lookup.lookup import cache_lookup
from rag.agent.nodes.cache_store.store import cache_store
from rag.agent.nodes.generate.direct_answer import direct_answer
from rag.agent.nodes.generate.generate import generate
from rag.agent.nodes.generate.no_results import no_results
from rag.agent.nodes.query.query import handle_query
from rag.agent.nodes.recall.recall import recall
from rag.agent.nodes.recall_fuse.fuse import recall_fuse
from rag.agent.nodes.recall_memory.memory import recall_memory
from rag.agent.nodes.rerank.rerank import rerank
from rag.agent.nodes.dynamic_topk.topk import dynamic_topk
from rag.agent.nodes.neighbor_expand.expand import neighbor_expand
from rag.agent.nodes.parent_expand.expand import parent_expand
from rag.agent.type import ContextSchema, MyState
from rag.config import get_settings


def _route_after_query(state: MyState) -> str:
    """条件边：范围外直接大模型兜底，范围内走召回。"""
    if state.get("is_out_of_scope"):
        return "direct_answer"
    return "recall"


def _route_after_cache(state: MyState):
    """条件边:缓存命中直达记忆写入(答案已流式下发);
    未命中按 [主查询]+sub_queries 扇出 Send 并行召回(开关关闭时恒单分支)。
    仅主查询分支携带 entities,避免多分支用同一实体集重复图召回。
    """
    if state.get("cache_hit"):
        return "add_memory"
    main_query = state.get("rewrite_query") or state["raw_query"]
    branches = [Send("recall", {
        "sub_query": main_query,
        "entities": state.get("query_entities") or [],
    })]
    if get_settings().QUERY_DECOMPOSITION_ENABLED:
        branches += [
            Send("recall", {"sub_query": sq, "entities": []})
            for sq in state.get("sub_queries") or []
        ]
    return branches


def _route_after_topk(state: MyState) -> str:
    """条件边：动态截断后为空时返回兜底话术，有结果才走生成。"""
    if state.get("recall_vec_results"):
        return "generate"
    return "no_results"


# 构建状态图
builder = StateGraph(MyState, context_schema=ContextSchema)

# 召回记忆
builder.add_node("recall_memory", recall_memory)
# 查询改写 + 范围判断
builder.add_node("handle_query", handle_query)
# 语义缓存查询(handle_query 之后,命中跳过检索链与生成)
builder.add_node("cache_lookup", cache_lookup)
# 语义缓存回写(generate 之后,best-effort)
builder.add_node("cache_store", cache_store)
# 知识库召回
builder.add_node("recall", recall)
# Send 分支 fan-in:跨子查询融合去重
builder.add_node("recall_fuse", recall_fuse)
# Sentence Window：召回后拉取相邻 chunk 扩展上下文
builder.add_node("neighbor_expand", neighbor_expand)
# 语义重排序
builder.add_node("rerank", rerank)
# 动态 top-k 截断
builder.add_node("dynamic_topk", dynamic_topk)
# Parent-Child Retrieval：chunk → 完整父文档展开
builder.add_node("parent_expand", parent_expand)
# 基于知识库生成
builder.add_node("generate", generate)
# 范围外直接大模型回答
builder.add_node("direct_answer", direct_answer)
# 记忆持久化（仅 generate 路由到此处，direct_answer / no_results 不写记忆）
builder.add_node("add_memory", add_memory)
# 召回为空时的兜底话术（不调 LLM）
builder.add_node("no_results", no_results)

#                                        ┌─ out-of-scope -> direct_answer ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
# START -> recall_memory -> handle_query ┤                                                                                                                                                               END
#                                        └─ in-scope -> cache_lookup ┬─ 命中 -> add_memory ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                                                    └─ 未命中 -> [Send x N] recall -> recall_fuse -> neighbor_expand -> rerank -> dynamic_topk -> parent_expand ┬─ generate -> cache_store -> add_memory ─┘
#                                                                                                                                                                              └─ no_results ────────────────────────────┘
builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_conditional_edges(
    "handle_query",
    _route_after_query,
    {"recall": "cache_lookup", "direct_answer": "direct_answer"},
)
builder.add_conditional_edges(
    "cache_lookup",
    _route_after_cache,
    {"add_memory": "add_memory", "recall": "recall"},
)
builder.add_edge("recall", "recall_fuse")
builder.add_edge("recall_fuse", "neighbor_expand")
builder.add_edge("neighbor_expand", "rerank")
builder.add_edge("rerank", "dynamic_topk")
builder.add_edge("dynamic_topk", "parent_expand")
builder.add_conditional_edges(
    "parent_expand",
    _route_after_topk,
    {"generate": "generate", "no_results": "no_results"},
)
builder.add_edge("generate", "cache_store")
builder.add_edge("cache_store", "add_memory")
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
        {
            "session_id": session_id,
            "raw_query": query,
            "is_out_of_scope": False,
            "sub_queries": [],
            "sub_recall_results": [],
        },
        context=context,
        stream_mode=["custom"],
        config=config,
    ):
        yield chunk
