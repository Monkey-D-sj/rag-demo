from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.prompts.generate import system_prompt


async def generate(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """基于知识库检索结果生成回答：逐 token 流式输出。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "生成回答中"))

    llm = runtime.context.llm
    chunks = state.get("recall_vec_results") or []

    # 结构化拼接上下文：每个 chunk 编号 + 带来源文档标题，同时构建引用元数据
    context_parts: list[str] = []
    citations: list[dict] = []
    for i, c in enumerate(chunks, 1):
        title = c.get("filename", "未知文档")
        text = c.get("text", "")
        context_parts.append(f"[{i}] (来源文档: {title})\n{text}")
        citations.append({
            "index": i,
            "text": text,
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
