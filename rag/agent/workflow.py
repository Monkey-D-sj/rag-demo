from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from rag.agent.nodes.agent.agent import agent_execute
from rag.agent.nodes.add_memory.memory import add_memory
from rag.agent.nodes.cache_lookup.lookup import cache_lookup
from rag.agent.nodes.cache_store.store import cache_store
from rag.agent.nodes.generate.context_answer import context_answer
from rag.agent.nodes.generate.generate import generate
from rag.agent.nodes.generate.no_results import no_results
from rag.agent.nodes.query.query import handle_query
from rag.agent.nodes.recall.recall import recall
from rag.agent.nodes.recall_fuse.fuse import recall_fuse
from rag.agent.nodes.recall_memory.memory import recall_memory
from rag.agent.nodes.rerank.rerank import rerank
from rag.agent.nodes.dynamic_topk.topk import dynamic_topk
from rag.agent.nodes.expand.expand import expand
from rag.agent.type import ContextSchema, MyState
from rag.config import get_settings


def _route_after_query(state: MyState) -> str:
    """条件边：会话上下文问题走 context_answer；范围外问题已由 handle_query 直接作答，直达 END；
    多步/交叉引用且开关开启时走 agent 分支，其余进缓存查找。"""
    if state.get("answer_from_context"):
        return "context_answer"
    if state.get("is_out_of_scope"):
        return "end"
    if get_settings().AGENT_MODE_ENABLED and state.get("needs_agent"):
        return "agent"
    return "recall"


def _route_after_agent(state: MyState) -> str:
    """只有明确收敛且证据非空才生成；其他退出原因均降级正常召回链。"""
    if state.get("agent_succeeded") and state.get("recall_vec_results"):
        return "generate"
    return "cache_lookup"


def _route_after_cache(state: MyState):
    """条件边:缓存命中直达记忆写入(答案已流式下发);
    未命中按 [主查询]+sub_queries 扇出 Send 并行召回(开关关闭时恒单分支)。
    """
    if state.get("cache_hit"):
        return "add_memory"
    main_query = state.get("rewrite_query") or state["raw_query"]
    branches = [Send("recall", {"sub_query": main_query})]
    if get_settings().QUERY_DECOMPOSITION_ENABLED:
        branches += [
            Send("recall", {"sub_query": sq})
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
# agent 多步工具检索(仅 needs_agent 且开关开启时进入)
builder.add_node("agent_execute", agent_execute)
# 语义缓存查询(handle_query 之后,命中跳过检索链与生成)
builder.add_node("cache_lookup", cache_lookup)
# 语义缓存回写(generate 之后,best-effort)
builder.add_node("cache_store", cache_store)
# 知识库召回
builder.add_node("recall", recall)
# Send 分支 fan-in:跨子查询融合去重
builder.add_node("recall_fuse", recall_fuse)
# 统一扩展（sentence window / parent-child 按 KB 动态分流）
builder.add_node("expand", expand)
# 语义重排序
builder.add_node("rerank", rerank)
# 动态 top-k 截断
builder.add_node("dynamic_topk", dynamic_topk)
# 基于知识库生成
builder.add_node("generate", generate)
# 仅依据会话上下文回答，不走知识库召回
builder.add_node("context_answer", context_answer)
# 记忆持久化（generate / context_answer 路由到此处；no_results 直达 END 不写记忆）
builder.add_node("add_memory", add_memory)
# 召回为空时的兜底话术（不调 LLM）
builder.add_node("no_results", no_results)

#                                        ┌─ context-only -> context_answer -> add_memory ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                                        ├─ out-of-scope -> END(handle_query 内直接作答) ────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
# START -> recall_memory -> handle_query ┤                                                                                                                                                              END
#                                        └─ in-scope -> cache_lookup ┬─ 命中 -> add_memory ────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                                                    └─ 未命中 -> [Send x N] recall -> recall_fuse -> rerank -> dynamic_topk -> expand ┬─ generate -> cache_store -> add_memory ─┘
#                                                                                                                                                      └─ no_results ───────────────────────────┘
builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_conditional_edges(
    "handle_query",
    _route_after_query,
    {
        "recall": "cache_lookup",
        "end": END,
        "context_answer": "context_answer",
        "agent": "agent_execute",
    },
)
builder.add_conditional_edges(
    "agent_execute",
    _route_after_agent,
    {"generate": "generate", "cache_lookup": "cache_lookup"},
)
builder.add_conditional_edges(
    "cache_lookup",
    _route_after_cache,
    {"add_memory": "add_memory", "recall": "recall"},
)
builder.add_edge("recall", "recall_fuse")
builder.add_edge("recall_fuse", "rerank")
builder.add_edge("rerank", "dynamic_topk")
builder.add_edge("dynamic_topk", "expand")
builder.add_conditional_edges(
    "expand",
    _route_after_topk,
    {"generate": "generate", "no_results": "no_results"},
)
builder.add_edge("generate", "cache_store")
builder.add_edge("cache_store", "add_memory")
builder.add_edge("context_answer", "add_memory")
builder.add_edge("add_memory", END)
builder.add_edge("no_results", END)

graph = builder.compile()


def build_initial_state(session_id: str, query: str) -> MyState:
    """构造生产与评测共用的初始状态。"""
    return {
        "session_id": session_id,
        "raw_query": query,
        "is_out_of_scope": False,
        "answer_from_context": False,
        "sub_queries": [],
        "sub_recall_results": [],
        "needs_agent": False,
        "agent_succeeded": False,
        "agent_skip_cache": False,
    }


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
        build_initial_state(session_id, query),
        context=context,
        stream_mode=["custom"],
        config=config,
    ):
        yield chunk
