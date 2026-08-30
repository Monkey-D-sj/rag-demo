from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.common.logging import get_logger
from rag.agent.type import ContextSchema, StreamEventType, stream_event

logger = get_logger()


async def recall(state: dict, runtime: Runtime[ContextSchema]) -> dict:
    """单分支知识库召回:接收 Send 负载 {sub_query}。

    结果以单元素列表返回,经 sub_recall_results 的 operator.add reducer 与
    其他并行分支拼接,recall_fuse 统一融合。分支内任何异常必须兜住:
    Send 分支抛异常会 fail 整个 run。
    """
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "检索知识库中..."))

    retriever = runtime.context.retriever
    if retriever is None:
        logger.warning("retriever 未注入,跳过知识库召回")
        return {"sub_recall_results": [[]]}

    try:
        # kb_ids 不传 → 搜全部知识库
        rows = await retriever.search(state["sub_query"])
    except Exception:  # noqa: BLE001 - 分支级容错,单分支失败不拖垮整图
        logger.warning("子查询召回失败,该分支降级为空", exc_info=True)
        rows = []
    return {"sub_recall_results": [rows]}
