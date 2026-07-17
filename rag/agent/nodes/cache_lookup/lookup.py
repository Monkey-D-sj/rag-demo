from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.config import get_settings

logger = get_logger()


async def cache_lookup(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """语义缓存查询:命中则直接下发缓存答案,路由跳过检索链与生成。

    开关关闭或未注入时透传;lookup 内部已吞异常,失败即未命中。
    """
    settings = get_settings()
    cache = runtime.context.semantic_cache
    if not settings.SEMANTIC_CACHE_ENABLED or cache is None:
        return state

    query = state.get("rewrite_query") or state["raw_query"]
    hit = await cache.lookup(query, session_id=state.get("session_id"))
    if hit is None:
        return state

    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "缓存命中"))
    writer(stream_event(StreamEventType.MESSAGE, hit["answer"]))
    # 复放原次回答的引用元数据,与 generate 节点的下发格式一致
    writer({"type": "citations", "data": hit["citations"]})

    state["generated"] = hit["answer"]
    state["citations"] = hit["citations"]
    state["cache_hit"] = True
    return state
