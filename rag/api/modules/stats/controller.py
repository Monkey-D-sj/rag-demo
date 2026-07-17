from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Query, Request

from rag.api.modules.stats import service

stats_router = APIRouter(prefix="/stats")


@stats_router.get("/llm/summary")
async def llm_summary(
    request: Request,
    start: datetime | None = None,
    end: datetime | None = None,
    group_by: Literal["day", "model", "source"] = "day",
):
    """LLM 调用统计聚合。默认最近 14 天。"""
    resolved_end = end or datetime.now(timezone.utc)
    resolved_start = start or resolved_end - timedelta(days=14)
    return await service.llm_summary(
        request.app.state.pg, resolved_start, resolved_end, group_by
    )


@stats_router.get("/llm/recent")
async def llm_recent(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    """最近调用明细(排查用)。"""
    return await service.llm_recent(request.app.state.pg, limit)
