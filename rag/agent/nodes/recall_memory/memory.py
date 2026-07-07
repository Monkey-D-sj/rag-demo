from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from rag.agent.type import ContextSchema, MessageRole, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger

logger = get_logger()


def _format_msg(r: dict) -> str:
    """单条消息格式化：有角色标签时输出 "用户：xxx"，否则裸文本兜底。"""
    text = r.get("text", "")
    meta = r.get("metadata")
    role = meta.get("role") if isinstance(meta, dict) else None
    label = MessageRole.display_of(role)
    return f"{label}：{text}" if label else text


async def recall_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从长期和短期记忆中召回相关内容，按角色区分用户/AI。

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

        # 合并:短片优先,按 raw text 去重,有 role 则加角色标签
        seen: set[str] = set()
        parts: list[str] = []
        for r in short_results + long_results:
            text = r.get("text", "")
            if not text or text in seen:
                continue
            seen.add(text)
            parts.append(_format_msg(r))

        state["context"] = "\n".join(parts)
    except Exception:
        # 记忆召回是辅助功能,失败不阻断流程——降级为空上下文,
        # handle_query 在无上下文时会透传原始查询。
        logger.warning("记忆检索失败,降级为空上下文继续", exc_info=True)
        writer(stream_event(StreamEventType.STATUS, "记忆检索暂时不可用"))
        state["context"] = ""

    return state
