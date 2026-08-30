import psycopg
import pytest

from rag.config import get_settings


@pytest.mark.integration
def test_table_and_extensions_exist():
    s = get_settings()
    with psycopg.connect(s.PG_ASYNC_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.long_term_memories')"
            )
            assert cur.fetchone()[0] == "long_term_memories"
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            assert cur.fetchone() is not None


@pytest.mark.integration
def test_document_tables_exist_with_novel_regulation_kbs():
    s = get_settings()
    with psycopg.connect(s.PG_ASYNC_DSN) as conn:
        with conn.cursor() as cur:
            for table in ("knowledge_bases", "documents", "document_chunks"):
                cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                assert cur.fetchone()[0] == table
            # 不比较顺序:中文 name 的排序依赖 DB collation,各环境不一致
            cur.execute("SELECT name FROM knowledge_bases")
            rows = cur.fetchall()
            assert {r[0] for r in rows} == {"书籍文献", "法规"}


@pytest.mark.integration
def test_semantic_cache_table_exists():
    s = get_settings()
    with psycopg.connect(s.PG_ASYNC_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.semantic_cache')")
            assert cur.fetchone()[0] == "semantic_cache"
            cur.execute(
                "SELECT indexname FROM pg_indexes"
                " WHERE tablename = 'semantic_cache'"
                " AND indexname = 'idx_semantic_cache_session_created_at'"
            )
            assert cur.fetchone() is not None
            cur.execute(
                "SELECT column_name, is_nullable, data_type"
                " FROM information_schema.columns"
                " WHERE table_name = 'semantic_cache'"
                " AND column_name = 'session_id'"
            )
            assert cur.fetchone() == ("session_id", "NO", "uuid")
            cur.execute(
                "SELECT indexname FROM pg_indexes"
                " WHERE tablename = 'semantic_cache'"
                " AND indexname = 'idx_semantic_cache_embedding'"
            )
            assert cur.fetchone() is None
