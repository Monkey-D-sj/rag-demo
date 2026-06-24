import psycopg
import pytest

from rag.config import get_settings


@pytest.mark.integration
def test_table_and_extensions_exist():
    s = get_settings()
    with psycopg.connect(s.pg_async_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.long_term_memories')"
            )
            assert cur.fetchone()[0] == "long_term_memories"
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            assert cur.fetchone() is not None


@pytest.mark.integration
def test_document_tables_exist_with_default_kb():
    s = get_settings()
    with psycopg.connect(s.pg_async_dsn) as conn:
        with conn.cursor() as cur:
            for table in ("knowledge_bases", "documents", "document_chunks"):
                cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                assert cur.fetchone()[0] == table
            cur.execute(
                "SELECT name FROM knowledge_bases "
                "WHERE id = '00000000-0000-0000-0000-000000000001'"
            )
            assert cur.fetchone()[0] == "default"
