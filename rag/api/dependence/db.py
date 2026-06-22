import redis.asyncio as redis
from fastapi import FastAPI
from psycopg_pool import AsyncConnectionPool


def get_pg(app: FastAPI) -> AsyncConnectionPool:
	return app.state.pg


def get_redis(app: FastAPI) -> redis.Redis:
	return app.state.redis
