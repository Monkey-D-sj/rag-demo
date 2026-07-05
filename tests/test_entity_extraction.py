from rag.document.entity_extraction import Entity, Relationship


def test_entity_type_outside_taxonomy_coerced_to_other():
    assert Entity(name="X", type="Superhero").type == "其他"


def test_entity_type_in_taxonomy_kept():
    assert Entity(name="X", type="Person").type == "Person"


def test_entity_name_and_endpoints_stripped():
    assert Entity(name=" 孙悟空 ").name == "孙悟空"
    r = Relationship(source=" A ", target=" B ")
    assert (r.source, r.target) == ("A", "B")
