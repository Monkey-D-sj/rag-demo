from langgraph.config import get_stream_writer

from rag.memory import memory_manager
from rag.type import MyState


def recall_memory(state: MyState) -> MyState:
	"""从记忆中召回与查询相关的内容"""
	writer = get_stream_writer()
	writer("检索记忆中...")

	results = memory_manager.search(state["raw_query"])
	state["recall_memory_results"] = results
	return state
