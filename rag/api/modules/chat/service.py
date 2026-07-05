"""Chat 流式服务 —— 对接 agent workflow 与 ChatStream SDK。"""

import asyncio
from collections.abc import AsyncIterator

from rag.api.common.stream import ChatStream
from rag.agent.type import ContextSchema
from rag.agent.workflow import invoke
from rag.common.logging import bind_session, get_logger, reset_session
from rag.document.retriever import KnowledgeRetriever
from rag.memory import MemoryManager
from rag.models.base import ChatModel

logger = get_logger()


async def stream_chat(
    session_id: str,
    query: str,
    *,
    llm: ChatModel,
    memory_manager: MemoryManager,
    retriever: KnowledgeRetriever,
) -> AsyncIterator[str]:
    """把工作流事件通过 ChatStream 编码为 SSE 行下发。

    生产者-消费者模式：后台 task 将 LangGraph 事件喂入 ChatStream，
    主协程从 ChatStream 出队 SSE 行并 yield 给 StreamingResponse。
    """
    token = bind_session(session_id)
    try:
        context = ContextSchema(
            llm=llm, memory_manager=memory_manager, retriever=retriever
        )
        stream = ChatStream()

        async def _produce() -> None:
            try:
                async for event in invoke(session_id, query, context):
                    await stream.send_event(event)
            except Exception as e:  # noqa: BLE001
                logger.exception("chat stream failed")
                await stream.error(str(e))
            finally:
                stream.close()

        task = asyncio.create_task(_produce())

        async for sse_line in stream:
            yield sse_line

        await task  # 确保生产者异常不被静默吞掉
    finally:
        reset_session(token)
