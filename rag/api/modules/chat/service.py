import json
from collections.abc import AsyncIterator

from rag.common.logging import get_logger
from rag.document.retriever import KnowledgeRetriever
from rag.memory import MemoryManager
from rag.models.base import ChatModel
from rag.agent.type import ContextSchema
from rag.agent.workflow import invoke

logger = get_logger()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def stream_chat(
    session_id: str,
    query: str,
    *,
    llm: ChatModel,
    memory_manager: MemoryManager,
    retriever: KnowledgeRetriever,
) -> AsyncIterator[str]:
    """把工作流事件逐条编码为 SSE 行下发。

    流已开始后 HTTP 状态码无法再改,故运行期异常以 error 事件帧告知前端,
    再补 [DONE] 收尾;客户端断开导致的 CancelledError 不拦截,交由上层处理。
    """
    context = ContextSchema(
        llm=llm, memory_manager=memory_manager, retriever=retriever
    )
    try:
        async for event in invoke(session_id, query, context):
            yield _sse(event)
    except Exception as e:  # noqa: BLE001 - 兜底转 error 帧
        logger.exception("chat stream failed")
        yield _sse({"type": "error", "data": str(e)})
    yield "data: [DONE]\n\n"
