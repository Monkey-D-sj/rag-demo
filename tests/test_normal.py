from rag.config import Settings
from rag.models.normal import NormalModel


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    async def ainvoke(self, messages):
        return _FakeMsg("answer")


async def test_ainvoke_returns_content():
    m = NormalModel(Settings())
    m._model = _FakeModel()
    assert await m.ainvoke(["hi"]) == "answer"
