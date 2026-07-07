from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger

logger = get_logger()


async def rerank(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """对 recall 召回的 chunk 做语义重排序，提升 top 位精度。

    未注入 reranker 时透传原始结果，不影响检索链路。
    """
    reranker = runtime.context.reranker
    if reranker is None:
        return state

    chunks = state.get("recall_vec_results")
    if not chunks:
        return state

    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "重排序中"))

    query = state.get("rewrite_query") or state["raw_query"]
    top_k = len(chunks)  # 保留全部，只调顺序不截断
    state["recall_vec_results"] = await reranker.rerank(query, chunks, top_k)
    return state
