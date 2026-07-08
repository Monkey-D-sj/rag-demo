from langgraph.config import get_stream_writer

from rag.agent.type import MyState, StreamEventType, stream_event

_NO_RESULT_MSG = """
未在当前知识库中找到能够直接回答您问题的依据。
建议您补充更多信息后重新提问，我将继续为您检索相关内容。
"""


async def no_results(state: MyState) -> MyState:
    """知识库无召回结果时直接返回兜底话术，不调 LLM、不写记忆。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "未找到相关内容"))
    writer(stream_event(StreamEventType.MESSAGE, _NO_RESULT_MSG))
    state["generated"] = _NO_RESULT_MSG
    return state
