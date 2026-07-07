from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from rag.agent.type import MyState, ContextSchema, StreamEventType, stream_event
from rag.common.logging import get_logger

logger = get_logger()


class QueryRewriteOutput(BaseModel):
    """查询改写 + 范围判断的结构化输出。"""

    rewrite_query: str = Field(
        description="改写后的独立查询文本；如果 is_out_of_scope 为 true，返回原始查询原文"
    )
    is_out_of_scope: bool = Field(
        description="查询是否与知识库无关（闲聊、编程、通用常识等可直接由大模型回答的问题）"
    )


system_prompt = """
你是一个查询分析助手，同时负责两项任务：**范围判断**和**查询改写**。

## 任务一：范围判断（is_out_of_scope）

判断用户的查询是否与知识库相关、需要检索知识库才能准确回答。

以下类型应标记为 **知识库无关**（is_out_of_scope = true），可直接由大模型自身知识回答：
- **闲聊问候**："你好"、"今天天气怎么样"、"讲个笑话"
- **编程/技术问题**："帮我写个排序算法"、"Python 怎么读取 CSV"
- **通用常识**：与知识库领域无关的百科问题（如"地球有多大"、"咖啡因的作用"）
- **实时/个人**："现在几点"、"我今天吃了什么"、"你是谁"
- **指令/动作**："帮我翻译这段话"、"总结一下我说的话"

以下类型应标记为 **知识库相关**（is_out_of_scope = false）：
- 需要查阅文档、书籍、资料才能准确回答的问题
- 涉及特定领域/专业知识的问题
- 用户明确要求基于知识库/文档回答的问题
- 如果拿不准，默认标记为知识库相关（宁可多检索，不要漏掉）

## 任务二：查询改写（rewrite_query）

当 is_out_of_scope = false 时，将用户的原始查询改写为适合知识库检索的独立查询：
1. **指代消解**：将上下文中的代词（他/她/它/他们/那个/这个/这里等）替换为具体的实体名称。
2. **省略补全**：如果用户查询省略了主语或关键信息，从上下文中提取并补全。
3. **保持原意**：不要添加用户没问的内容，只补全指代和省略信息。

当 is_out_of_scope = true 时，rewrite_query 直接返回原始查询原文。

## 输出格式

仅输出一个 JSON 对象，包含 rewrite_query（字符串）和 is_out_of_scope（布尔值）两个字段。
不要包含任何其他文字，也不要用代码块包裹。

## 示例

上下文：用户刚才在问刘备的结拜兄弟有哪些。
用户查询：他三弟是谁
→ {"rewrite_query": "刘备的三弟是谁", "is_out_of_scope": false}

上下文：空或无关。
用户查询：孙悟空为什么被压在五指山下
→ {"rewrite_query": "孙悟空为什么被压在五指山下", "is_out_of_scope": false}

上下文：空。
用户查询：你好啊
→ {"rewrite_query": "你好啊", "is_out_of_scope": true}

上下文：空。
用户查询：帮我用 Python 写一个快速排序
→ {"rewrite_query": "帮我用 Python 写一个快速排序", "is_out_of_scope": true}

上下文：空。
用户查询：今天天气真不错
→ {"rewrite_query": "今天天气真不错", "is_out_of_scope": true}
"""


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

    return state
