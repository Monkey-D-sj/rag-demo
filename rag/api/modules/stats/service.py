"""llm_call_log 聚合查询。group_by 白名单映射,无字符串拼接注入面。"""
from datetime import datetime

from rag.db.postgres import get_cursor

# group_by → bucket 表达式(白名单,防注入)
_BUCKET_EXPRS = {
    "day": "date_trunc('day', created_at)",
    "model": "model",
    "source": "source",
}

_RECENT_MAX_LIMIT = 200


async def llm_summary(pool, start: datetime, end: datetime, group_by: str) -> dict:
    bucket_expr = _BUCKET_EXPRS.get(group_by)
    if bucket_expr is None:
        raise ValueError(f"不支持的 group_by: {group_by}")
    sql = f"""
        SELECT {bucket_expr} AS bucket,
               count(*)::int                                    AS calls,
               count(*) FILTER (WHERE status = 'success')::int  AS success,
               count(*) FILTER (WHERE status = 'failed')::int   AS failed,
               count(*) FILTER (WHERE status = 'rejected')::int AS rejected,
               coalesce(sum(input_tokens), 0)::bigint           AS input_tokens,
               coalesce(sum(output_tokens), 0)::bigint          AS output_tokens,
               coalesce(sum(cost), 0)                           AS cost,
               round(avg(latency_ms))::int                      AS avg_latency_ms,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
        FROM llm_call_log
        WHERE created_at >= %(start)s AND created_at < %(end)s
        GROUP BY bucket
        ORDER BY bucket
    """
    async with get_cursor(pool) as cur:
        await cur.execute(sql, {"start": start, "end": end})
        rows = await cur.fetchall()
    return {"group_by": group_by, "rows": rows}


async def llm_recent(pool, limit: int) -> dict:
    clamped = max(1, min(limit, _RECENT_MAX_LIMIT))
    sql = """
        SELECT id, created_at, call_type, model, source, session_id, status,
               error_type, attempts, latency_ms, input_tokens, output_tokens, cost
        FROM llm_call_log
        ORDER BY id DESC
        LIMIT %(limit)s
    """
    async with get_cursor(pool) as cur:
        await cur.execute(sql, {"limit": clamped})
        rows = await cur.fetchall()
    return {"rows": rows}
