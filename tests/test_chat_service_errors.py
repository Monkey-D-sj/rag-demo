"""chat producer 的异常必须映射为友好文案,原始异常不外泄到 SSE。"""
import json

from rag.api.modules.chat import service
from rag.common.exception import CircuitOpenError


class _MockRewriteResult:
    """使 handle_query 成功返回 is_out_of_scope=True,路由到 direct_answer 触发 LLM 错误。"""
    rewrite_query = "你好"
    is_out_of_scope = True
    entities = []


class _ExplodingLLM:
    """astream 抛熔断异常的假 LLM; ainvoke_structured 正常返回以便路由到 direct_answer。"""

    async def ainvoke_structured(self, messages, schema):
        return _MockRewriteResult()

    def astream(self, messages):
        async def _gen():
            raise CircuitOpenError("gov:cb:chat open")
            yield  # pragma: no cover
        return _gen()


class _NoopMemory:
    async def search(self, session_id, query, top_k=5, filters=None):
        return []

    async def get_recent_messages(self, session_id, n=10):
        return []

    async def add_message(self, session_id, text, metadata=None):
        pass

    async def persist_turn(self, session_id, query, answer):
        pass


class _NoopRetriever:
    async def search(self, query, knowledge_base_ids=None, top_k=5):
        return []

    async def fetch_parent_contents(self, document_ids):
        return {}


async def test_stream_error_event_is_friendly_not_raw():
    lines = []
    async for sse in service.stream_chat(
        "s1", "你好",
        llm=_ExplodingLLM(), memory_manager=_NoopMemory(),
        retriever=_NoopRetriever(), reranker=None, pool=None,
    ):
        lines.append(sse)
    error_events = [
        json.loads(l.removeprefix("data: "))
        for l in lines
        if l.startswith("data: {") and '"error"' in l
    ]
    assert error_events, "应产生 error 事件"
    assert error_events[0]["data"] == "AI 服务暂时不可用,请稍后重试"
    assert "gov:cb" not in error_events[0]["data"]  # 原始异常信息不外泄
