from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.common.logging import get_logger
from rag.config import get_settings

logger = get_logger()


async def cache_store(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """把本轮生成的答案回写语义缓存。store 内部已吞异常,best-effort。"""
    settings = get_settings()
    cache = runtime.context.semantic_cache
    if not settings.SEMANTIC_CACHE_ENABLED or cache is None:
        return state

    answer = state.get("generated")
    if not answer:
        return state

    session_id = state.get("session_id")
    if not session_id:
        logger.warning("语义缓存回写跳过：缺少 session_id")
        return state

    query = state.get("rewrite_query") or state["raw_query"]
    await cache.store(
        query,
        answer,
        state.get("citations") or [],
        session_id=session_id,
    )
    return state
