import sys
import types
import unittest
from unittest.mock import Mock, patch


def _install_test_stubs() -> None:
    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        sys.modules["dotenv"] = dotenv

    if "psycopg2" not in sys.modules:
        psycopg2 = types.ModuleType("psycopg2")
        psycopg2_extras = types.ModuleType("psycopg2.extras")
        psycopg2_pool = types.ModuleType("psycopg2.pool")

        class Json:
            def __init__(self, adapted):
                self.adapted = adapted

        class RealDictCursor:
            pass

        class SimpleConnectionPool:
            def __init__(self, *args, **kwargs):
                pass

        psycopg2_extras.Json = Json
        psycopg2_extras.RealDictCursor = RealDictCursor
        psycopg2_pool.SimpleConnectionPool = SimpleConnectionPool
        psycopg2.extras = psycopg2_extras
        psycopg2.pool = psycopg2_pool

        sys.modules["psycopg2"] = psycopg2
        sys.modules["psycopg2.extras"] = psycopg2_extras
        sys.modules["psycopg2.pool"] = psycopg2_pool

    if "redis" not in sys.modules:
        redis_module = types.ModuleType("redis")

        class ConnectionPool:
            def __init__(self, *args, **kwargs):
                pass

        class Redis:
            def __init__(self, *args, **kwargs):
                pass

        redis_module.ConnectionPool = ConnectionPool
        redis_module.Redis = Redis
        sys.modules["redis"] = redis_module

    if "openai" not in sys.modules:
        openai_module = types.ModuleType("openai")

        class OpenAI:
            def __init__(self, *args, **kwargs):
                pass

        openai_module.OpenAI = OpenAI
        sys.modules["openai"] = openai_module


_install_test_stubs()

from psycopg2.extras import Json

from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory
from rag.memory.adapters.short_term_redis import RedisShortTermMemory
from rag.memory.manager import MemoryManager


class FakeLongTermMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict | None]] = []

    def add(self, session_id: str, text: str, metadata: dict | None = None) -> str:
        return "memory-id"

    def search(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        self.calls.append((query, top_k, filters))
        return [{"text": "match"}]

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        return []

    def update(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        return None

    def delete(self, memory_id: str) -> None:
        return None

    def get_by_time(self, start: float, end: float) -> list[dict]:
        return []


class MemoryModuleTests(unittest.TestCase):
    def test_memory_manager_search_forwards_filters(self) -> None:
        long_term = FakeLongTermMemory()
        manager = MemoryManager(long_term=long_term)

        result = manager.search("hello", top_k=3, filters={"session_id": "s1"})

        self.assertEqual(result, [{"text": "match"}])
        self.assertEqual(long_term.calls, [("hello", 3, {"session_id": "s1"})])

    def test_short_term_get_recent_zero_returns_empty(self) -> None:
        adapter = RedisShortTermMemory.__new__(RedisShortTermMemory)
        adapter._redis = Mock()

        result = adapter.get_recent("session-1", 0)

        self.assertEqual(result, [])
        adapter._redis.lrange.assert_not_called()

    @patch("rag.memory.adapters.long_term_pgsql.generate_embeddings_batch")
    @patch("rag.memory.adapters.long_term_pgsql.get_cursor")
    def test_long_term_add_wraps_metadata_as_json(
        self, mock_get_cursor: Mock, mock_embeddings: Mock
    ) -> None:
        mock_embeddings.return_value = [[0.1, 0.2]]
        cur = Mock()
        mock_get_cursor.return_value.__enter__.return_value = cur
        adapter = PgVectorLongTermMemory.__new__(PgVectorLongTermMemory)

        adapter.add("session-1", "hello", {"topic": "faq"})

        params = cur.execute.call_args.args[1]
        self.assertIsInstance(params["metadata"], Json)
        self.assertEqual(
            params["metadata"].adapted,
            {"session_id": "session-1", "topic": "faq"},
        )

    @patch("rag.memory.adapters.long_term_pgsql.generate_embeddings_batch")
    @patch("rag.memory.adapters.long_term_pgsql.get_cursor")
    def test_long_term_update_merges_metadata_instead_of_replacing(
        self, mock_get_cursor: Mock, mock_embeddings: Mock
    ) -> None:
        mock_embeddings.return_value = [[0.1, 0.2]]
        cur = Mock()
        mock_get_cursor.return_value.__enter__.return_value = cur
        adapter = PgVectorLongTermMemory.__new__(PgVectorLongTermMemory)

        adapter.update("memory-1", "updated", {"topic": "billing"})

        sql, params = cur.execute.call_args.args
        self.assertIn("|| %(metadata)s::jsonb", sql)
        self.assertIsInstance(params["metadata"], Json)
        self.assertEqual(params["metadata"].adapted, {"topic": "billing"})


if __name__ == "__main__":
    unittest.main()
