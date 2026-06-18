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
