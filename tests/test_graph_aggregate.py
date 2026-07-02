from rag.document.entity_extraction import Entity, ExtractionResult, Relationship
from rag.graph.aggregate import aggregate


def _res(entities, rels):
    return ExtractionResult(entities=entities, relationships=rels)


def test_merges_same_name_across_chunks_unions_chunk_ids():
    c0 = _res([Entity("孙悟空", "Person", "d")], [])
    c1 = _res([Entity("孙悟空", "Person", "d2")], [])
    ents, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(ents) == 1
    assert ents[0]["name"] == "孙悟空"
    assert sorted(ents[0]["chunk_ids"]) == ["doc:0", "doc:1"]


def test_type_conflict_uses_majority_then_first():
    c0 = _res([Entity("X", "Person", "d")], [])
    c1 = _res([Entity("X", "Creature", "d")], [])
    c2 = _res([Entity("X", "Person", "d")], [])
    ents, _ = aggregate([("doc:0", c0), ("doc:1", c1), ("doc:2", c2)])
    assert ents[0]["type"] == "Person"  # 众数


def test_relation_direction_normalized_and_keywords_unioned():
    c0 = _res(
        [Entity("唐僧", "Person", ""), Entity("孙悟空", "Person", "")],
        [Relationship("孙悟空", "唐僧", "师徒", "")],
    )
    c1 = _res(
        [Entity("唐僧", "Person", ""), Entity("孙悟空", "Person", "")],
        [Relationship("唐僧", "孙悟空", "取经", "")],
    )
    _, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(rels) == 1
    r = rels[0]
    assert (r["source"], r["target"]) == ("唐僧", "孙悟空")  # 字典序规范化
    assert sorted(r["keywords"]) == ["取经", "师徒"]


def test_drops_relation_with_unknown_endpoint():
    c0 = _res([Entity("A", "Person", "")], [Relationship("A", "B", "k", "")])
    ents, rels = aggregate([("doc:0", c0)])
    assert [e["name"] for e in ents] == ["A"]
    assert rels == []
