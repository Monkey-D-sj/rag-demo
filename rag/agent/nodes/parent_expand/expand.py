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
    """Parent-Child Retrieval：对法规 KB chunk 按 doc_id 补查全文替换 text。

    关闭时透传；补查失败或无 retriever 时降级为仅去重（保留原始 text）。
    """
    settings = get_settings()

    if not settings.PARENT_CHILD_ENABLED:
        return state

    chunks = state.get("recall_vec_results")
    if not isinstance(chunks, list) or not chunks:
        return state

    # 收集需补查的法规文档 ID
    expand_ids: set[str] = set()
    for c in chunks:
        if _should_expand(c) and c.get("document_id"):
            expand_ids.add(str(c["document_id"]))

    # 补查全文
    contents: dict[str, str] = {}
    if expand_ids:
        retriever = getattr(runtime.context, "retriever", None)
        if retriever is not None:
            try:
                contents = await retriever.fetch_parent_contents(list(expand_ids))
            except Exception:
                logger.warning("补查父文档全文失败，降级去重", exc_info=True)

    # 按 document_id 分组去重，法规 KB 有全文时替换 text
    groups: dict[str, dict] = {}
    orphans: list[dict] = []
    for c in chunks:
        doc_id = str(c.get("document_id", ""))
        if not doc_id:
            orphans.append(c)
            continue

        if _should_expand(c) and doc_id in contents:
            # 法规 KB + 补查到全文：替换 text
            if doc_id not in groups:
                groups[doc_id] = dict(c)
                groups[doc_id]["text"] = contents[doc_id]
            else:
                existing_score = groups[doc_id].get("rerank_score", 0)
                this_score = c.get("rerank_score", 0)
                if this_score > existing_score:
                    groups[doc_id] = dict(c)
                    groups[doc_id]["text"] = contents[doc_id]
        else:
            # 非法规 KB 或补查无结果：仅去重
            if doc_id not in groups:
                groups[doc_id] = c
            else:
                existing_score = groups[doc_id].get("rerank_score", 0)
                this_score = c.get("rerank_score", 0)
                if this_score > existing_score:
                    groups[doc_id] = c

    expanded: list[dict] = list(groups.values())
    expanded.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    orphans.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    expanded.extend(orphans)

    state["recall_vec_results"] = expanded
    logger.debug("parent_expand: %d chunks -> %d docs", len(chunks), len(expanded))
    return state
