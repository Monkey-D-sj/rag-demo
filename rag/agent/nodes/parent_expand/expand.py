from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.common.logging import get_logger
from rag.config import get_settings
from rag.document import REGULATION_KB_ID

logger = get_logger()


def _should_expand(chunk: dict) -> bool:
    """判断 chunk 是否属于应展开为父文档的 KB。"""
    return str(chunk.get("knowledge_base_id", "")) == REGULATION_KB_ID


async def parent_expand(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """Parent-Child Retrieval：对政策法规 KB 的 chunk 按 document_id 去重，替换为完整父文档。

    关闭时透传；非法规 KB 或缺少 document_content 时保持原样。
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

        # 非法规 KB → 透传，但按文档去重
        if not _should_expand(c):
            if doc_id not in groups:
                groups[doc_id] = c
            else:
                existing_score = groups[doc_id].get("rerank_score", 0)
                this_score = c.get("rerank_score", 0)
                if this_score > existing_score:
                    groups[doc_id] = c
            continue

        content = c.get("document_content")
        if not content:
            if doc_id not in groups:
                groups[doc_id] = c
            else:
                existing_score = groups[doc_id].get("rerank_score", 0)
                this_score = c.get("rerank_score", 0)
                if this_score > existing_score:
                    groups[doc_id] = c
            continue

        # 法规 KB + 有父文档内容：替换 text 为全文
        if doc_id not in groups:
            groups[doc_id] = dict(c)
            groups[doc_id]["text"] = content
            groups[doc_id].pop("document_content", None)
        else:
            existing_score = groups[doc_id].get("rerank_score", 0)
            this_score = c.get("rerank_score", 0)
            if this_score > existing_score:
                groups[doc_id] = dict(c)
                groups[doc_id]["text"] = content
                groups[doc_id].pop("document_content", None)

    # 重建结果列表，保持 rerank_score 降序
    expanded: list[dict] = list(groups.values())
    expanded.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    orphans.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    expanded.extend(orphans)

    state["recall_vec_results"] = expanded

    logger.debug("parent_expand: %d chunks -> %d docs", len(chunks), len(expanded))
    return state
