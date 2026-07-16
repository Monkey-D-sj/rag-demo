from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.common.logging import get_logger
from rag.config import get_settings

logger = get_logger()


async def parent_expand(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """Parent-Child Retrieval：按 document_id 去重，用完整父文档替换 chunk 文本。

    关闭时透传；chunk 缺少 document_content 时保持原样。
    """
    settings = get_settings()

    if not settings.PARENT_CHILD_ENABLED:
        return state

    chunks = state.get("recall_vec_results")
    if not isinstance(chunks, list) or not chunks:
        return state

    # 按 document_id 分组，每组保留 rerank_score 最高的一条
    groups: dict[str, dict] = {}
    orphans: list[dict] = []
    for c in chunks:
        doc_id = str(c.get("document_id", ""))
        if not doc_id:
            orphans.append(c)
            continue

        content = c.get("document_content")
        if not content:
            # 无父文档内容，透传（按文档分组但保留原始 text）
            if doc_id not in groups:
                groups[doc_id] = c
            else:
                existing_score = groups[doc_id].get("rerank_score", 0)
                this_score = c.get("rerank_score", 0)
                if this_score > existing_score:
                    groups[doc_id] = c
            continue

        # 有父文档内容：取 rerank_score 最高的 chunk 作为代表，替换 text 为全文
        if doc_id not in groups:
            groups[doc_id] = dict(c)
            groups[doc_id]["text"] = content
        else:
            existing_score = groups[doc_id].get("rerank_score", 0)
            this_score = c.get("rerank_score", 0)
            if this_score > existing_score:
                groups[doc_id] = dict(c)
                groups[doc_id]["text"] = content

    # 重建结果列表，保持 rerank_score 降序
    expanded: list[dict] = list(groups.values())
    expanded.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    # orphans 追加在末尾
    orphans.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    expanded.extend(orphans)

    state["recall_vec_results"] = expanded

    logger.debug("parent_expand: %d chunks -> %d parent docs", len(chunks), len(expanded))
    return state
