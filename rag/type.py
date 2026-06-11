from typing import TypedDict


class MyState(TypedDict):
	session_id: str

	# ----------- 检索 -----------
	raw_query: str
	rewrite_query: str

	# ----------- 召回 -----------
	recall_bm25_results: list[dict]
	recall_vec_results: list[dict]
	recall_memory_results: list[dict]

	# ----------- 生成 -----------
	generated: str
