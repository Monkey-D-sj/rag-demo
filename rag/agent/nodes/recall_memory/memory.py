from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from rag.agent.type import MyState, ContextSchema, StreamEventType, stream_event
from rag.common.logging import get_logger

logger = get_logger()


async def recall_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从长期和短期记忆中召回相关内容。

    记忆是辅助性的——失败时降级为空上下文继续,不阻断主流程。
    """

    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "检索记忆中..."))

    try:
        memory_manager = runtime.context.memory_manager

        # 长期记忆:向量检索(适配器内部已按 session_id 过滤)
        long_results = await memory_manager.search(
            state["session_id"], state["raw_query"]
        )

        # 短期记忆:最近会话消息
        short_results = await memory_manager.get_recent_messages(state["session_id"])

        # 合并为上下文文本,短时优先,去重
        seen: set[str] = set()
        parts: list[str] = []
        for r in short_results:
            text = r.get("text", "")
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
        for r in long_results:
            text = r.get("text", "")
            if text and text not in seen:
                seen.add(text)
                parts.append(text)

        state["context"] = "\n".join(parts)
    except Exception:
        # 记忆召回是辅助功能,失败不阻断流程——降级为空上下文,
        # handle_query 在无上下文时会透传原始查询。
        logger.warning("记忆检索失败,降级为空上下文继续", exc_info=True)
        writer(stream_event(StreamEventType.STATUS, "记忆检索暂时不可用"))
        state["context"] = ""

    return state
