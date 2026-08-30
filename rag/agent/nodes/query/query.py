from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from rag.agent.type import MyState, ContextSchema, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.prompts.generate import direct_system_prompt
from rag.prompts.query import system_prompt

logger = get_logger()


class QueryRewriteOutput(BaseModel):
    """查询改写、范围判断和会话上下文问答路由的结构化输出。"""

    rewrite_query: str = Field(
        description="改写后的独立查询文本；如果 is_out_of_scope 为 true，返回原始查询原文"
    )
    is_out_of_scope: bool = Field(
        description="查询是否与知识库无关（闲聊、编程、通用常识等可直接由大模型回答的问题）"
    )
    answer_from_context: bool = Field(
        default=False,
        description=(
            "用户是否在询问本次会话之前说过的内容或对话历史（如‘我第一次讲了啥’、"
            "‘刚才我说了什么’）；为 true 时只依据上下文回答，不进行知识库检索"
        ),
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="复杂问题拆解出的独立子查询(每个自包含,实体显式,无指代);简单问题或 is_out_of_scope 为 true 时返回空数组",
    )


async def handle_query(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """查询分析：范围判断 + 改写 + 拆解；范围外查询直接流式作答。

    使用结构化输出（json_mode）。范围外问题不检索、不写回记忆，在此直接回答。
    """
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "深度思考中"))

    llm = runtime.context.llm

    try:
        result: QueryRewriteOutput = await llm.ainvoke_structured(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"""
用户查询: {state["raw_query"]}
上下文: {state["context"]}
"""
                ),
            ],
            QueryRewriteOutput,
        )
        state["is_out_of_scope"] = result.is_out_of_scope
        # 短路命中时按契约钳为固定值,不信任 LLM 对被跳过任务(二/三/四)的输出
        state["rewrite_query"] = (
            state["raw_query"] if result.is_out_of_scope else result.rewrite_query
        )
        # 兼容旧的结构化结果对象：未提供该字段时按普通查询处理。
        state["answer_from_context"] = (
            getattr(result, "answer_from_context", False)
            and not result.is_out_of_scope
        )
        # 上限 3 个,防 LLM 超量输出;短路命中时钳空;开关关闭时路由侧不消费,此处不做 gate
        state["sub_queries"] = [] if result.is_out_of_scope else result.sub_queries[:3]
        if state["sub_queries"] and not result.is_out_of_scope:
            writer(stream_event(
                StreamEventType.STATUS, f"已拆解为 {len(state['sub_queries'])} 个子问题"
            ))

        if result.is_out_of_scope:
            logger.info(
                "查询判定为知识库范围外，跳过改写，直接大模型兜底",
                extra={"query": state["raw_query"][:200]},
            )
    except Exception:
        # 结构化输出失败时降级：默认走检索路径，rewrite_query 回退原始查询
        logger.warning("查询分析结构化输出失败，降级为透传原始查询", exc_info=True)
        state["rewrite_query"] = state["raw_query"]
        state["is_out_of_scope"] = False
        state["answer_from_context"] = False
        state["sub_queries"] = []
        return state

    # 范围外直接作答：不检索、不写回记忆。流式放在 try/except 之外，
    # 避免 astream 中途失败被误当作结构化输出失败而降级进 RAG 检索链。
    if state["is_out_of_scope"]:
        writer(stream_event(StreamEventType.STATUS, "理解问题中"))
        writer(stream_event(StreamEventType.STATUS, "生成回答中"))
        messages = [
            SystemMessage(content=direct_system_prompt),
            HumanMessage(content=state["raw_query"]),
        ]
        parts: list[str] = []
        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", chunk)
            if not token:
                continue
            parts.append(token)
            writer(stream_event(StreamEventType.MESSAGE, token))
        state["generated"] = "".join(parts)

    return state
