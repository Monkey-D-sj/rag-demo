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
from rag.observability.langfuse import get_callback_handler, observe_root, session_scope

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

        handler = get_callback_handler()
        # metadata 里的 langfuse_session_id 仍保留:LangChain 回调链根节点
        # 自己也会解析这个 key 并做一次 session 传播,双重设置对同一条 trace
        # 无副作用,属于防御性冗余(即便 session_scope 的传播机制变化也不丢失)
        config = (
            {"callbacks": [handler], "metadata": {"langfuse_session_id": session_id}}
            if handler
            else None
        )

        async def _run_traced() -> None:
            # with(非 async with)包住:session_scope 只做同步 contextvar 读写,
            # 不阻塞事件循环。本函数若被 observe_root 包装,则持有 trace 根 span,
            # CallbackHandler 与 retriever 的 @observe span 均继承环境 OTel 上下文,
            # 嵌套进同一条 trace,而不是各自另起一条独立顶层 trace
            with session_scope(session_id):
                async for event in invoke(session_id, query, context, config=config):
                    await stream.send_event(event)

        if handler is not None:
            _run_traced = observe_root(name="chat")(_run_traced)

        async def _produce() -> None:
            try:
                await _run_traced()
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
        try:
            reset_session(token)
        except ValueError:
            # 生成器被弃置时可能在异 Context 中终结,reset 失败仅影响该条日志,忽略
            logger.debug("reset_session 跳过: 生成器在异 Context 中终结")
