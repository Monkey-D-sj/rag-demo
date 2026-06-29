import redis.asyncio as redis

from rag.config import Settings


def create_redis_client(settings: Settings) -> redis.Redis:
    """创建异步 Redis 客户端（连接池复用，解码为 str）。"""
    pool = redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        max_connections=settings.REDIS_MAX_CONNECTIONS,
        decode_responses=True,
    )
    return redis.Redis(connection_pool=pool)
