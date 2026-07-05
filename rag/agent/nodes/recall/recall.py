from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.common.logging import get_logger
from rag.document import DEFAULT_KB_ID
from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event

logger = get_logger()


async def recall(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从知识库召回相关 chunk(向量检索),写入 recall_vec_results。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "检索知识库中..."))

    retriever = runtime.context.retriever
    if retriever is None:
        logger.warning("retriever 未注入,跳过知识库召回")
        state["recall_vec_results"] = []
        return state

    # 改写后的查询更适合检索,缺失时回退原始查询
    query = state.get("rewrite_query") or state["raw_query"]
    # rewrite 节点当前被禁用,此日志持续暴露「改写从未生效」的事实
    logger.info(
        "知识库召回使用%s查询",
        "改写后" if state.get("rewrite_query") else "原始",
        extra={"query": query[:200]},
    )
    state["recall_vec_results"] = await retriever.search(query, DEFAULT_KB_ID)
    return state
