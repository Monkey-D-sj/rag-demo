from types import SimpleNamespace

import rag.db.neo4j as n


def test_create_driver_passes_uri_and_auth(monkeypatch):
    captured = {}

    class _FakeDriver:
        pass

    def fake_driver(uri, auth=None):
        captured["uri"] = uri
        captured["auth"] = auth
        return _FakeDriver()

    monkeypatch.setattr(n.AsyncGraphDatabase, "driver", staticmethod(fake_driver))
    settings = SimpleNamespace(
        NEO4J_URI="bolt://x:7687", NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw", NEO4J_DATABASE="neo4j",
    )
    drv = n.create_neo4j_driver(settings)
    assert isinstance(drv, _FakeDriver)
    assert captured["uri"] == "bolt://x:7687"
    assert captured["auth"] == ("neo4j", "pw")


def test_db_package_exports_neo4j_helpers():
    import rag.db as db
    assert hasattr(db, "create_neo4j_driver")
    assert hasattr(db, "ensure_graph_constraints")
