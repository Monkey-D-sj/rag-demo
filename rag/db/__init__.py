from rag.db.redis import get_redis_client
from rag.db.postgres import get_pg_pool, ensure_pgvector_extension

redis_client = get_redis_client()
