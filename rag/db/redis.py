import os

import redis
from dotenv import load_dotenv

load_dotenv()

_pool: redis.ConnectionPool | None = None


def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端（连接池复用）"""
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None,
            max_connections=10,
            decode_responses=True,
        )
    return redis.Redis(connection_pool=_pool)
