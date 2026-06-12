import os
from contextlib import contextmanager

from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

load_dotenv()

_pool: SimpleConnectionPool | None = None

MIN_CONN = 2
MAX_CONN = 10


def get_pg_pool() -> SimpleConnectionPool:
    """获取 PostgreSQL 连接池"""
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            MIN_CONN,
            MAX_CONN,
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            dbname=os.getenv("PG_DATABASE", "rag_memory"),
            user=os.getenv("PG_USER", "rag"),
            password=os.getenv("PG_PASSWORD", "rag123"),
        )
    return _pool


@contextmanager
def get_cursor():
    """获取游标的上下文管理器，自动归还连接"""
    pool = get_pg_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def ensure_pgvector_extension() -> None:
    """确保 pgvector 扩展已启用"""
    with get_cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")


def ensure_pgbm25_extension() -> None:
    """确保 pg_bm25 扩展已启用（需要 ParadeDB 镜像）"""
    with get_cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_bm25")


def bm25_search(
    table: str,
    query: str,
    columns: list[str] | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    """BM25 全文检索

    Args:
        table: 表名
        query: 搜索查询文本
        columns: 要搜索的列名列表，默认所有文本列
        limit: 返回条数
        offset: 偏移量
    """
    col_list = columns or ["text"]
    cols = ", ".join(col_list)

    sql = f"""
        SELECT *, paradedb.score(id) AS bm25_score
        FROM {table}
        WHERE {table} @@@ paradedb.parse(%(query)s)
        ORDER BY bm25_score DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """

    with get_cursor() as cur:
        cur.execute(sql, {"query": query, "limit": limit, "offset": offset})
        return cur.fetchall()


def hybrid_search(
    table: str,
    query: str,
    embedding_column: str = "embedding",
    limit: int = 10,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> list[dict]:
    """混合检索：BM25 + 向量相似度加权融合

    Args:
        table: 表名
        query: 搜索查询文本
        embedding_column: 向量列名
        limit: 返回条数
        bm25_weight: BM25 权重
        vector_weight: 向量权重
    """
    # 先生成 query embedding
    from rag.models.embedding import generate_embeddings_batch
    query_emb = generate_embeddings_batch([query])[0]

    sql = f"""
        SELECT *,
               COALESCE(paradedb.score(id), 0) * %(bm25_w)s
               + (1 - ({embedding_column} <=> %(embedding)s)) * %(vec_w)s
               AS hybrid_score
        FROM {table}
        WHERE {table} @@@ paradedb.parse(%(query)s)
           OR {embedding_column} IS NOT NULL
        ORDER BY hybrid_score DESC
        LIMIT %(limit)s
    """

    with get_cursor() as cur:
        cur.execute(sql, {
            "query": query,
            "embedding": query_emb,
            "limit": limit,
            "bm25_w": bm25_weight,
            "vec_w": vector_weight,
        })
        return cur.fetchall()
