from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from rag.agent.type import MyState, ContextSchema, StreamEventType, stream_event


async def recall_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从长期和短期记忆中召回相关内容"""

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

        # 合并为上下文文本
        parts: list[str] = []
        for r in short_results:
            parts.append(r.get("text", ""))
        for r in long_results:
            parts.append(r.get("text", ""))

        state["context"] = "\n".join(parts)
        return state
    except Exception as e:
        writer(stream_event(StreamEventType.STATUS, f"记忆检索失败: {e}"))
        raise
