from rag.document.entity_extraction import Entity, ExtractionResult, Relationship
from rag.graph.aggregate import aggregate


def _res(entities, rels):
    return ExtractionResult(entities=entities, relationships=rels)


def test_merges_same_name_across_chunks_unions_chunk_ids():
    c0 = _res([Entity(name="孙悟空", type="Person", description="d")], [])
    c1 = _res([Entity(name="孙悟空", type="Person", description="d2")], [])
    ents, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(ents) == 1
    assert ents[0]["name"] == "孙悟空"
    assert sorted(ents[0]["chunk_ids"]) == ["doc:0", "doc:1"]


def test_type_conflict_uses_majority_then_first():
    c0 = _res([Entity(name="X", type="Person", description="d")], [])
    c1 = _res([Entity(name="X", type="Creature", description="d")], [])
    c2 = _res([Entity(name="X", type="Person", description="d")], [])
    ents, _ = aggregate([("doc:0", c0), ("doc:1", c1), ("doc:2", c2)])
    assert ents[0]["type"] == "Person"  # 众数


def test_relation_direction_normalized_and_keywords_unioned():
    c0 = _res(
        [Entity(name="唐僧", type="Person"), Entity(name="孙悟空", type="Person")],
        [Relationship(source="孙悟空", target="唐僧", keywords="师徒")],
    )
    c1 = _res(
        [Entity(name="唐僧", type="Person"), Entity(name="孙悟空", type="Person")],
        [Relationship(source="唐僧", target="孙悟空", keywords="取经")],
    )
    _, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(rels) == 1
    r = rels[0]
    assert (r["source"], r["target"]) == ("唐僧", "孙悟空")  # 字典序规范化
    assert sorted(r["keywords"]) == ["取经", "师徒"]


def test_drops_relation_with_unknown_endpoint():
    c0 = _res(
        [Entity(name="A", type="Person")],
        [Relationship(source="A", target="B", keywords="k")],
    )
    ents, rels = aggregate([("doc:0", c0)])
    assert [e["name"] for e in ents] == ["A"]
    assert rels == []
