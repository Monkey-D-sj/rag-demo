from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import MyState, ContextSchema

system_prompt = """
你是一个专业的关务助手, 你的任务是根据用户的查询, 提供专业的关务信息.
有不确定的地方，例如：他/那么。
从上下文获取信息，改写消息返回
"""


async def handle_query(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
	"""处理查询:改写为中间步骤,整体 ainvoke(不逐 token 流式)"""
	# ----------- 输出 -----------
	writer = get_stream_writer()
	writer({"type": "status", "data": "深度思考中"})

	llm = runtime.context.llm

	state["rewrite_query"] = await llm.ainvoke([
		SystemMessage(content=system_prompt),
		HumanMessage(content=f"""
用户查询: {state["raw_query"]}
上下文: {state["context"]}
"""),
	])
	return state
