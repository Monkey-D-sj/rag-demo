import redis.asyncio as redis
from fastapi import Request
from psycopg_pool import AsyncConnectionPool


def get_pg(request: Request) -> AsyncConnectionPool:
    return request.app.state.pg


def get_redis(request: Request) -> redis.Redis:
    return request.app.state.redis
