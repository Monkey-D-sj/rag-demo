import pytest

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
