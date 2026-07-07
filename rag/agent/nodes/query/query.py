from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import MyState, ContextSchema, StreamEventType, stream_event

system_prompt = """
你是一个查询改写助手。你的任务是将用户的原始查询改写为适合知识库检索的独立查询。

## 改写规则

1. **指代消解**：将上下文中的代词（他/她/它/他们/那个/这个/这里等）替换为具体的实体名称。
2. **省略补全**：如果用户查询省略了主语或关键信息，从上下文中提取并补全。
3. **保持原意**：不要添加用户没问的内容，只补全指代和省略信息。
4. **直接输出**：只输出改写后的查询文本，不要加引号、解释或前缀。

## 示例

上下文：用户刚才在问刘备的结拜兄弟有哪些。
用户查询：他三弟是谁
改写：刘备的三弟是谁

上下文：用户刚才在问花果山的位置。
用户查询：那里有什么著名的猴子
改写：花果山有什么著名的猴子

上下文：空或无关。
用户查询：孙悟空为什么被压在五指山下
改写：孙悟空为什么被压在五指山下

## 兜底

如果上下文为空、无关，或查询本身已经完整独立，直接返回原始查询，不做改动。
"""


async def handle_query(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
	"""处理查询:改写为中间步骤,整体 ainvoke(不逐 token 流式)"""
	# ----------- 输出 -----------
	writer = get_stream_writer()
	writer(stream_event(StreamEventType.STATUS, "深度思考中"))

	llm = runtime.context.llm

	state["rewrite_query"] = await llm.ainvoke([
		SystemMessage(content=system_prompt),
		HumanMessage(content=f"""
用户查询: {state["raw_query"]}
上下文: {state["context"]}
"""),
	])
	return state
