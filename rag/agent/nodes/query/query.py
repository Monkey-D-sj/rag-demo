from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from rag.agent.type import MyState, ContextSchema, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.prompts.query import system_prompt

logger = get_logger()


class QueryRewriteOutput(BaseModel):
    """查询改写 + 范围判断的结构化输出。"""

    rewrite_query: str = Field(
        description="改写后的独立查询文本；如果 is_out_of_scope 为 true，返回原始查询原文"
    )
    is_out_of_scope: bool = Field(
        description="查询是否与知识库无关（闲聊、编程、通用常识等可直接由大模型回答的问题）"
    )
    entities: list[str] = Field(
        default_factory=list,
        description="查询中出现的专有名词实体(人名/地名/物名等);is_out_of_scope 为 true 或查询无实体时返回空数组",
    )


async def handle_query(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """查询分析：判断是否在知识库范围内 + 改写查询。使用结构化输出（json_mode）。"""
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
        state["rewrite_query"] = result.rewrite_query
        state["is_out_of_scope"] = result.is_out_of_scope
        state["query_entities"] = result.entities

        if result.is_out_of_scope:
            logger.info(
                "查询判定为知识库范围外，跳过改写，直接大模型兜底",
                extra={"query": state["raw_query"][:200]},
            )
            writer(stream_event(StreamEventType.STATUS, "理解问题中"))
    except Exception:
        # 结构化输出失败时降级：默认走检索路径，rewrite_query 回退原始查询
        logger.warning("查询分析结构化输出失败，降级为透传原始查询", exc_info=True)
        state["rewrite_query"] = state["raw_query"]
        state["is_out_of_scope"] = False
        state["query_entities"] = []

    return state
