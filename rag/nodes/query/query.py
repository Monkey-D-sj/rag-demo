from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer

from rag.models.normal import llm_invoke
from rag.type import MyState

system_prompt = """
你是一个专业的关务助手, 你的任务是根据用户的查询, 提供专业的关务信息.
如果有不确定的地方，例如：他/那么。
从上下文获取信息，改写消息返回
"""


def handle_query(state: MyState) -> MyState:
	"""处理查询"""
	# ----------- 输出 -----------
	writer = get_stream_writer()
	writer("深度思考中")
	
	state["rewrite_query"] = llm_invoke([
		SystemMessage(content=system_prompt),
		HumanMessage(content=f"""
用户查询: {state["raw_query"]}
上下文: {state["raw_query"]}
"""),
	])
	return state
