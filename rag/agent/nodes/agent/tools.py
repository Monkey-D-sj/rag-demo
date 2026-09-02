"""agent 分支工具:检索 / 取文档全文 / 查会话记忆。

每个工具拆成"纯 handler(返回 list[dict] 行)"与"给人看的渲染文本"两层:
- executor(agent 节点)直接调 handler 拿结构化行喂 recall_vec_results,不解析 ToolMessage;
- 模型看到的 observation 是 _render 的文本,两处不互相依赖,避免逻辑漂移。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from rag.agent.type import ContextSchema
from rag.common.logging import get_logger

logger = get_logger()

DOCUMENT_WINDOW_DEFAULT_CHARS = 4_000
DOCUMENT_WINDOW_MAX_CHARS = 8_000
TOOL_OBSERVATION_ROW_CHARS = 4_000
TOOL_OBSERVATION_TOTAL_CHARS = 12_000


# ── handler:返回统一行形 {text, filename, metadata, document_id, chunk_index?} ──

def _normalize_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "text": text,
            "filename": r.get("filename") or "知识库文档",
            "metadata": r.get("metadata") or {},
            "document_id": str(r.get("document_id") or ""),
            "chunk_index": r.get("chunk_index"),
        })
    return out


async def _kb_search(
    context: ContextSchema, session_id: str, query: str, top_k: int = 5,
) -> list[dict]:
    """知识库语义检索:返回与 query 最相关的片段行。query 需自包含、无指代。"""
    if context.retriever is None:
        return []
    try:
        top_k = min(max(int(top_k), 1), 10)
    except (TypeError, ValueError):
        top_k = 5
    try:
        rows = await context.retriever.search(query, top_k=top_k)
    except Exception:  # noqa: BLE001 - 工具级容错,空结果比中断好
        logger.warning("agent 工具 retrieve_kb 失败，返回空", exc_info=True)
        return []
    return _normalize_rows(rows)


async def _doc_fetch(
    context: ContextSchema,
    session_id: str,
    document_id: str,
    offset: int = 0,
    max_chars: int = DOCUMENT_WINDOW_DEFAULT_CHARS,
) -> list[dict]:
    """按文档 ID 取一个有界文本窗口；长文档通过 offset 分页读取。"""
    if context.retriever is None or not document_id:
        return []
    try:
        offset = max(int(offset), 0)
    except (TypeError, ValueError):
        offset = 0
    try:
        max_chars = min(max(int(max_chars), 500), DOCUMENT_WINDOW_MAX_CHARS)
    except (TypeError, ValueError):
        max_chars = DOCUMENT_WINDOW_DEFAULT_CHARS
    try:
        contents = await context.retriever.fetch_parent_contents([str(document_id)])
    except Exception:  # noqa: BLE001
        logger.warning("agent 工具 fetch_document 失败，返回空", exc_info=True)
        return []
    full_text = (contents or {}).get(str(document_id), "")
    if not full_text or offset >= len(full_text):
        return []
    end = min(offset + max_chars, len(full_text))
    text = full_text[offset:end].strip()
    if not text:
        return []
    return [{
        "text": text,
        "filename": str(document_id),
        "metadata": {
            "document_offset": offset,
            "document_end": end,
            "document_total_chars": len(full_text),
        },
        "document_id": str(document_id),
        "chunk_index": None,
    }]


async def _mem_search(
    context: ContextSchema, session_id: str, query: str,
) -> list[dict]:
    """当前会话记忆中检索与 query 相关的历史内容。"""
    if context.memory_manager is None:
        return []
    try:
        rows = await context.memory_manager.search(session_id, query, top_k=3)
    except Exception:  # noqa: BLE001
        logger.warning("agent 工具 search_memory 失败，返回空", exc_info=True)
        return []
    out = []
    for r in rows or []:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "text": text,
            "filename": "会话记忆",
            "metadata": r.get("metadata") or {},
            "document_id": "",
            "chunk_index": None,
        })
    return out


async def _finish_collection(
    context: ContextSchema, session_id: str,
) -> list[dict]:
    """完成标志 handler；是否已有证据由执行器统一校验。"""
    return []


# ── 渲染:模型看到的 observation ──

def _render(rows: list[dict]) -> str:
    if not rows:
        return "未检索到相关内容。"
    parts = []
    for i, r in enumerate(rows, 1):
        title = r.get("filename") or "未知文档"
        text = (r.get("text") or "")[:TOOL_OBSERVATION_ROW_CHARS]
        attrs = []
        if r.get("document_id"):
            attrs.append(f"document_id={r['document_id']}")
        if r.get("chunk_index") is not None:
            attrs.append(f"chunk_index={r['chunk_index']}")
        meta = r.get("metadata") or {}
        if "document_total_chars" in meta:
            attrs.append(
                "字符范围="
                f"{meta.get('document_offset', 0)}-{meta.get('document_end', 0)}"
                f"/{meta['document_total_chars']}"
            )
        suffix = f" ({', '.join(attrs)})" if attrs else ""
        parts.append(f"[{i}] 《{title}》{suffix}\n{text}")
    rendered = "\n\n".join(parts)
    if len(rendered) > TOOL_OBSERVATION_TOTAL_CHARS:
        return rendered[:TOOL_OBSERVATION_TOTAL_CHARS] + "\n…工具结果已截断"
    return rendered


# ── 工具与注册表 ──

def build_tools(context: ContextSchema, session_id: str) -> list[BaseTool]:
    """构造闭包绑定运行时的工具列表,供 bind_tools。"""

    @tool
    async def retrieve_kb(query: str, top_k: int = 5) -> str:
        """在知识库中按语义检索与 query 最相关的片段。query 需自包含、无指代、显式写出实体。"""
        rows = await _kb_search(context, session_id, query=query, top_k=top_k)
        return _render(rows)

    @tool
    async def fetch_document(
        document_id: str,
        offset: int = 0,
        max_chars: int = DOCUMENT_WINDOW_DEFAULT_CHARS,
    ) -> str:
        """按文档 ID 获取有界原文窗口。ID 必须来自检索结果；长文档按返回的字符范围调整 offset 继续读取。"""
        rows = await _doc_fetch(
            context,
            session_id,
            document_id=document_id,
            offset=offset,
            max_chars=max_chars,
        )
        return _render(rows)

    @tool
    async def search_memory(query: str) -> str:
        """在当前会话记忆(历史对话)中检索与 query 相关的内容。"""
        rows = await _mem_search(context, session_id, query=query)
        return _render(rows)

    @tool
    async def finish_evidence_collection() -> str:
        """仅当工具已经返回回答所需的全部证据时调用，用于明确结束检索。"""
        return "证据收集完成。"

    return [
        retrieve_kb,
        fetch_document,
        search_memory,
        finish_evidence_collection,
    ]


HANDLERS = {
    "retrieve_kb": _kb_search,
    "fetch_document": _doc_fetch,
    "search_memory": _mem_search,
    "finish_evidence_collection": _finish_collection,
}
