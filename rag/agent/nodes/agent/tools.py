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
    context: ContextSchema, session_id: str, document_id: str,
) -> list[dict]:
    """按文档 ID 取全文(法规条款需看完整原文时用)。"""
    if context.retriever is None or not document_id:
        return []
    try:
        contents = await context.retriever.fetch_parent_contents([str(document_id)])
    except Exception:  # noqa: BLE001
        logger.warning("agent 工具 fetch_document 失败，返回空", exc_info=True)
        return []
    text = (contents or {}).get(str(document_id), "")
    if not text.strip():
        return []
    return [{
        "text": text,
        "filename": str(document_id),
        "metadata": {},
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


# ── 渲染:模型看到的 observation ──

def _render(rows: list[dict]) -> str:
    if not rows:
        return "未检索到相关内容。"
    parts = []
    for i, r in enumerate(rows, 1):
        title = r.get("filename") or "未知文档"
        text = (r.get("text") or "")[:400]
        parts.append(f"[{i}] 《{title}》\n{text}")
    return "\n\n".join(parts)


# ── 工具与注册表 ──

def build_tools(context: ContextSchema, session_id: str) -> list[BaseTool]:
    """构造闭包绑定运行时的工具列表,供 bind_tools。"""

    @tool
    async def retrieve_kb(query: str, top_k: int = 5) -> str:
        """在知识库中按语义检索与 query 最相关的片段。query 需自包含、无指代、显式写出实体。"""
        rows = await _kb_search(context, session_id, query=query, top_k=top_k)
        return _render(rows)

    @tool
    async def fetch_document(document_id: str) -> str:
        """按文档 ID 获取该文档完整原文(需核对条款/参照指向的完整上下文时用)。文档 ID 来自检索结果的 document_id。"""
        rows = await _doc_fetch(context, session_id, document_id=document_id)
        return _render(rows)

    @tool
    async def search_memory(query: str) -> str:
        """在当前会话记忆(历史对话)中检索与 query 相关的内容。"""
        rows = await _mem_search(context, session_id, query=query)
        return _render(rows)

    return [retrieve_kb, fetch_document, search_memory]


HANDLERS = {
    "retrieve_kb": _kb_search,
    "fetch_document": _doc_fetch,
    "search_memory": _mem_search,
}
