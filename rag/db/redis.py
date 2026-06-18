import redis.asyncio as redis

from rag.config import Settings


def create_redis_client(settings: Settings) -> redis.Redis:
    """创建异步 Redis 客户端（连接池复用，解码为 str）。"""
    pool = redis.ConnectionPool(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )
    return redis.Redis(connection_pool=pool)
