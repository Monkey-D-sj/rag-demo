from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from rag.type import MyState, ContextSchema


def recall_memory(state: MyState, runtime: Runtime) -> MyState:
    """从长期和短期记忆中召回相关内容"""

    writer = get_stream_writer()
    writer("检索记忆中...")
    
    try:
        memory_manager = runtime.context.memory_manager
    
        # 长期记忆：向量检索
        long_results = memory_manager.search(
            state["raw_query"], filters={"session_id": state["session_id"]}
        )
    
        # 短期记忆：最近会话消息
        short_results = memory_manager.get_recent_messages(state["session_id"])
    
        # 合并为上下文文本
        parts: list[str] = []
        for r in short_results:
            parts.append(r.get("text", ""))
        for r in long_results:
            parts.append(r.get("text", ""))
    
        state["context"] = "\n".join(parts)
        return state
    except Exception as e:
        writer(f"记忆检索失败: {e}")
        raise e
