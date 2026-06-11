from langgraph.config import get_stream_writer

from rag.models.normal import llm_invoke
from rag.type import MyState

system_prompt = """
你是一个专业的关务助手, 你的任务是根据用户的查询, 提供专业的关务信息.
"""


def handle_query(state: MyState) -> MyState:
	"""处理查询"""
	# ----------- 输出 -----------
	writer = get_stream_writer()
	writer("深度思考中")
	
	state["rewrite_query"] = llm_invoke(state["raw_query"])
	return state
