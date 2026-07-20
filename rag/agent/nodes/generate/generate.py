from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.prompts.generate import system_prompt


def _build_citation_label(filename: str, metadata: dict | None) -> str:
    """从 chunk metadata 构建结构化引用标签。

    示例：
      - 法规类：**《治安管理处罚法》第二十三条**
      - 书籍类：**《西游记》第一回**
      - 无结构化元数据时回退：**《xxx》**
    """
    meta = metadata or {}
    article = (meta.get("article") or "").strip()
    chapter = (meta.get("chapter") or "").strip()

    parts = [f"《{filename}》"]
    if chapter:
        parts.append(chapter)
    if article:
        parts.append(article)

    return f"**{' '.join(parts)}**"


async def generate(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """基于知识库检索结果生成回答：逐 token 流式输出。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "生成回答中"))

    llm = runtime.context.llm
    chunks = state.get("recall_vec_results") or []

    # 结构化拼接上下文：每个 chunk 编号 + 文档/章节/条款信息 + 正文
    context_parts: list[str] = []
    citations: list[dict] = []
    for i, c in enumerate(chunks, 1):
        title = c.get("filename", "未知文档")
        text = c.get("text", "")
        meta = c.get("metadata") or {}

        # 构建上下文行：含条款/章节信息供 LLM 引用
        article = (meta.get("article") or "").strip()
        chapter = (meta.get("chapter") or "").strip()
        locator = " ".join(p for p in [chapter, article] if p)
        header = f"[{i}] (来源: 《{title}》{(' ' + locator) if locator else ''})".rstrip()

        context_parts.append(f"{header}\n{text}")
        citations.append({
            "index": i,
            "text": _build_citation_label(title, meta),
            "snippet": text,
            "document_title": title,
        })

    knowledge = "\n\n".join(context_parts)
    state["citations"] = citations

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
查询: {state["raw_query"]}
知识库内容: {knowledge}
"""
        ),
    ]

    parts: list[str] = []
    async for chunk in llm.astream(messages):
        token = getattr(chunk, "content", chunk)
        if not token:
            continue
        parts.append(token)
        writer(stream_event(StreamEventType.MESSAGE, token))

    # 生成完成后下发引用元数据，供前端渲染可点击来源卡片
    writer({"type": "citations", "data": citations})

    state["generated"] = "".join(parts)
    return state
