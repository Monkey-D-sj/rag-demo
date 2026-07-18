from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.config import get_settings
from rag.document.retriever import fuse_multi_query_results


async def recall_fuse(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """Send 分支 fan-in:跨子查询二级 RRF 融合去重,写入 recall_vec_results。

    单分支时融合函数原样透传,行为与拆解前完全一致;
    全空时产出空列表,交由 _route_after_topk 走 no_results 兜底。
    """
    result_lists = state.get("sub_recall_results") or []
    state["recall_vec_results"] = fuse_multi_query_results(
        result_lists, rrf_k=get_settings().RETRIEVER_RRF_K
    )
    return state
