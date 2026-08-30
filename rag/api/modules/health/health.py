import asyncio
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
async def health(request: Request):
    """就绪探针: 检查 pg / redis / minio / rabbitmq / neo4j 连通性, 供容器编排使用。"""
    checks: dict[str, str] = {}
    healthy = True

    # ── PostgreSQL ──
    try:
        pg = request.app.state.pg
        async with pg.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        healthy = False

    # ── Redis ──
    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        healthy = False

    # ── MinIO ──
    try:
        await asyncio.to_thread(request.app.state.minio.list_buckets)
        checks["minio"] = "ok"
    except Exception as e:
        checks["minio"] = f"error: {e}"
        healthy = False

    # ── RabbitMQ ──
    try:
        publisher = request.app.state.task_publisher
        if publisher is None:
            checks["rabbitmq"] = "disabled"
        else:
            conn = publisher.connection
            if conn.is_closed or conn.reconnecting:
                raise RuntimeError("connection closed or reconnecting")
            # 真实 AMQP 往返: 删除不存在的随机队列(无副作用), 避开本地队列缓存
            await asyncio.wait_for(
                publisher.channel.queue_delete(f"_health_probe_{uuid.uuid4().hex}"),
                timeout=5,
            )
            checks["rabbitmq"] = "ok"
    except Exception as e:
        checks["rabbitmq"] = f"error: {e}"
        healthy = False

    # ── Neo4j ──
    neo4j = request.app.state.neo4j
    if neo4j is None:
        checks["neo4j"] = "disabled"
    else:
        try:
            await neo4j.verify_connectivity()
            checks["neo4j"] = "ok"
        except Exception as e:
            checks["neo4j"] = f"error: {e}"
            healthy = False

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
