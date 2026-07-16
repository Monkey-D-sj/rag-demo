from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.common.logging import get_logger
from rag.config import get_settings
from rag.document import store

logger = get_logger()


async def neighbor_expand(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """Sentence Window：对每个检索到的 chunk 拉取同文档相邻 ±N 个 chunk。

    关闭时透传；pool 未注入时跳过。
    """
    settings = get_settings()

    if not settings.SENTENCE_WINDOW_ENABLED:
        return state

    pool = runtime.context.pool
    if pool is None:
        logger.warning("pool 未注入 ContextSchema，跳过 sentence window")
        return state

    chunks = state.get("recall_vec_results")
    if not isinstance(chunks, list) or not chunks:
        return state

    window_size = settings.SENTENCE_WINDOW_SIZE
    max_multiplier = settings.SENTENCE_WINDOW_MAX_MULTIPLIER
    max_results = len(chunks) * max_multiplier

    # 按 (document_id, chunk_index) 去重
    seen: set[tuple[str, int]] = set()
    expanded: list[dict] = []

    for c in chunks:
        # 标记原始 chunk 为已见
        doc_id = str(c.get("document_id", ""))
        chunk_idx = c.get("chunk_index")
        if doc_id and isinstance(chunk_idx, int):
            seen.add((doc_id, chunk_idx))
        expanded.append(c)

    # 对每个原始 chunk 拉邻居
    for c in chunks:
        if len(expanded) >= max_results:
            break

        doc_id = str(c.get("document_id", ""))
        chunk_idx = c.get("chunk_index")
        if not doc_id or not isinstance(chunk_idx, int):
            continue

        try:
            neighbors = await store.get_neighbor_chunks(
                pool, doc_id, chunk_idx, window_size,
            )
        except Exception:
            logger.warning(
                "拉取邻居 chunk 失败 doc=%s idx=%s", doc_id, chunk_idx, exc_info=True,
            )
            continue

        base_score = c.get("rerank_score", 0)
        for n in neighbors:
            n_idx = n["chunk_index"]
            key = (doc_id, n_idx)
            if key in seen:
                continue
            if len(expanded) >= max_results:
                break
            seen.add(key)

            expanded.append({
                "id": None,
                "document_id": doc_id,
                "chunk_index": n_idx,
                "text": n["text"],
                "metadata": n.get("metadata", {}),
                "filename": c.get("filename", ""),
                "knowledge_base_id": c.get("knowledge_base_id", ""),
                "document_content": c.get("document_content"),
                "rerank_score": base_score,
                "sources": c.get("sources", []),
                "_neighbor": True,
            })

    state["recall_vec_results"] = expanded

    logger.debug("neighbor_expand: %d chunks -> %d (window=%d)", len(chunks), len(expanded), window_size)
    return state
