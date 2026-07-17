from rag.config import Settings
from rag.models.embedding import EmbeddingModel


class _FakeItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddings:
    def __init__(self, calls):
        self._calls = calls

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        # 故意乱序返回，验证按 index 排序
        return _FakeResp([_FakeItem(1, [0.2]), _FakeItem(0, [0.1])])


async def test_embed_orders_by_index():
    m = EmbeddingModel(Settings())
    calls = []
    m._client.embeddings = _FakeEmbeddings(calls)  # 注入 fake
    out = await m.embed(["a", "b"])
    assert out == [[0.1], [0.2]]
    assert calls[0]["input"] == ["a", "b"]


# ── 治理层织入 ──────────────────────────────────────────

from contextlib import asynccontextmanager

from rag.governance.guard import CallTracker


class _GuardSpy:
    def __init__(self):
        self.acquired: list[str] = []
        self.trackers: list[CallTracker] = []

    @asynccontextmanager
    async def acquire(self, quota):
        self.acquired.append(quota)
        yield

    @asynccontextmanager
    async def track(self, call_type, model):
        t = CallTracker(call_type=call_type, model=model)
        self.trackers.append(t)
        yield t


class _FakeUsage:
    prompt_tokens = 7


class _FakeEmbData:
    def __init__(self, index):
        self.index = index
        self.embedding = [0.0] * 4


class _FakeEmbResponse:
    data = [_FakeEmbData(0)]
    usage = _FakeUsage()


class _FakeEmbeddingsClient:
    class embeddings:  # noqa: N801 - 模仿 openai SDK 结构
        @staticmethod
        async def create(**kw):
            return _FakeEmbResponse()


async def test_embed_guard_weave_records_usage():
    guard = _GuardSpy()
    m = EmbeddingModel(Settings(), guard=guard)
    m._client = _FakeEmbeddingsClient()
    result = await m.embed(["文本"])
    assert len(result) == 1
    assert guard.acquired == ["embedding"]
    t = guard.trackers[0]
    assert (t.call_type, t.input_tokens) == ("embedding", 7)
