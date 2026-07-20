from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.common.logging import get_logger
from rag.config import get_settings
from rag.document import REGULATION_KB_ID, store

logger = get_logger()


def _is_regulation(chunk: dict) -> bool:
    """判断 chunk 是否属于法规 KB，应走 parent-child 展开。"""
    return str(chunk.get("knowledge_base_id", "")) == REGULATION_KB_ID


def _dedup_by_doc_id(chunks: list[dict]) -> list[dict]:
    """按 document_id 去重：同 doc 保留 rerank_score 最高的一条。"""
    groups: dict[str, dict] = {}
    orphans: list[dict] = []
    for c in chunks:
        doc_id = str(c.get("document_id", ""))
        if not doc_id:
            orphans.append(c)
            continue
        if doc_id not in groups or c.get("rerank_score", 0) > groups[doc_id].get("rerank_score", 0):
            groups[doc_id] = c
    return list(groups.values()) + orphans


async def expand(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """统一扩展节点：按 chunk 的 knowledge_base_id 动态选择策略。

    - 法规 KB：parent-child — 补查父文档全文替换 text，按 doc_id 去重
    - 普通 KB：sentence window — 拉取同文档 ±N 相邻 chunk 补充上下文
    - 混合结果：两条路径并行，最终按 rerank_score 合并排序
    """
    settings = get_settings()
    chunks: list[dict] = state.get("recall_vec_results") or []
    if not chunks:
        return state

    # ── 分流 ──
    regulation: list[dict] = []
    general: list[dict] = []
    for c in chunks:
        if _is_regulation(c):
            regulation.append(c)
        else:
            general.append(c)

    # ── 法规 KB：parent-child 全文展开 ──
    if regulation and settings.PARENT_CHILD_ENABLED:
        reg_expanded = await _expand_regulation(regulation, runtime)
    else:
        reg_expanded = regulation

    # ── 普通 KB：sentence window 邻居补全 ──
    if general and settings.SENTENCE_WINDOW_ENABLED:
        gen_expanded = await _expand_general(general, runtime, settings)
    else:
        gen_expanded = general

    # ── 合并 ──
    merged = reg_expanded + gen_expanded
    merged.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

    logger.debug(
        "expand: %d chunks → %d (regulation=%d, general=%d)",
        len(chunks), len(merged), len(reg_expanded), len(gen_expanded),
    )
    state["recall_vec_results"] = merged
    return state


# ═══════════════════════════════════════════════════════════════════
# 法规 KB：parent-child
# ═══════════════════════════════════════════════════════════════════

async def _expand_regulation(
    chunks: list[dict], runtime: Runtime[ContextSchema],
) -> list[dict]:
    """法规 KB：补查父文档全文，替换 text，按 doc_id 去重。"""
    # 收集需补查的文档 ID
    expand_ids: set[str] = set()
    for c in chunks:
        if c.get("document_id"):
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

    # 按 doc_id 分组去重，有全文时替换 text
    groups: dict[str, dict] = {}
    orphans: list[dict] = []
    for c in chunks:
        doc_id = str(c.get("document_id", ""))
        if not doc_id:
            orphans.append(c)
            continue

        if doc_id in contents:
            if doc_id not in groups:
                groups[doc_id] = dict(c)
                groups[doc_id]["text"] = contents[doc_id]
            else:
                if c.get("rerank_score", 0) > groups[doc_id].get("rerank_score", 0):
                    groups[doc_id] = dict(c)
                    groups[doc_id]["text"] = contents[doc_id]
        else:
            if doc_id not in groups or c.get("rerank_score", 0) > groups[doc_id].get("rerank_score", 0):
                groups[doc_id] = c

    expanded: list[dict] = list(groups.values())
    expanded.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    orphans.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    expanded.extend(orphans)
    return expanded


# ═══════════════════════════════════════════════════════════════════
# 普通 KB：sentence window
# ═══════════════════════════════════════════════════════════════════

async def _expand_general(
    chunks: list[dict], runtime: Runtime[ContextSchema], settings,
) -> list[dict]:
    """普通 KB：为每个 chunk 拉取同文档 ±N 相邻 chunk。"""
    pool = runtime.context.pool
    if pool is None:
        logger.warning("pool 未注入，跳过 sentence window")
        return chunks

    window_size = settings.SENTENCE_WINDOW_SIZE
    max_multiplier = settings.SENTENCE_WINDOW_MAX_MULTIPLIER
    max_results = len(chunks) * max_multiplier

    seen: set[tuple[str, int]] = set()
    expanded: list[dict] = []

    for c in chunks:
        doc_id = str(c.get("document_id", ""))
        chunk_idx = c.get("chunk_index")
        if doc_id and isinstance(chunk_idx, int):
            seen.add((doc_id, chunk_idx))
        expanded.append(c)

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

    return expanded
