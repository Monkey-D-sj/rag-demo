from rag.document.entity_extraction import (
    Entity,
    ExtractionResult,
    Relationship,
    extract_entities,
)


def test_entity_type_outside_taxonomy_coerced_to_other():
    assert Entity(name="X", type="Superhero").type == "其他"


def test_entity_type_in_taxonomy_kept():
    assert Entity(name="X", type="Person").type == "Person"


def test_entity_name_and_endpoints_stripped():
    assert Entity(name=" 孙悟空 ").name == "孙悟空"
    r = Relationship(source=" A ", target=" B ")
    assert (r.source, r.target) == ("A", "B")


class _FakeLLM:
    """只实现 ainvoke_structured 的鸭子类型 ChatModel。"""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    async def ainvoke_structured(self, messages, schema):
        assert schema is ExtractionResult
        self.calls += 1
        return self._result


async def test_empty_text_short_circuits_without_llm_call():
    llm = _FakeLLM(ExtractionResult())
    result = await extract_entities(llm, "   ")
    assert result == ExtractionResult()
    assert llm.calls == 0


async def test_relation_with_unknown_endpoint_filtered():
    llm = _FakeLLM(
        ExtractionResult(
            entities=[Entity(name="孙悟空", type="Person")],
            relationships=[
                Relationship(source="孙悟空", target="唐僧", keywords="师徒"),
            ],
        )
    )
    result = await extract_entities(llm, "text")
    assert [e.name for e in result.entities] == ["孙悟空"]
    assert result.relationships == []


async def test_happy_path_passes_through():
    llm = _FakeLLM(
        ExtractionResult(
            entities=[
                Entity(name="孙悟空", type="Person"),
                Entity(name="唐僧", type="Person"),
            ],
            relationships=[
                Relationship(source="孙悟空", target="唐僧", keywords="师徒"),
            ],
        )
    )
    result = await extract_entities(llm, "text")
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert llm.calls == 1


async def test_entity_missing_name_dropped_not_fatal():
    # 模型漏掉 name 字段时整个结果仍可解析,该记录被过滤而不是整 chunk 失败
    raw = ExtractionResult.model_validate(
        {
            "entities": [{"type": "Person"}, {"name": "唐僧", "type": "Person"}],
            "relationships": [],
        }
    )
    llm = _FakeLLM(raw)
    result = await extract_entities(llm, "text")
    assert [e.name for e in result.entities] == ["唐僧"]
