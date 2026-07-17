import asyncio
from dataclasses import dataclass

from rag.common.logging import get_logger
from rag.db.postgres import get_cursor
from rag.governance.pricing import compute_cost

logger = get_logger()

_INSERT_SQL = """
INSERT INTO llm_call_log
    (call_type, model, source, session_id, status, error_type,
     attempts, latency_ms, input_tokens, output_tokens, cost)
VALUES
    (%(call_type)s, %(model)s, %(source)s, %(session_id)s, %(status)s, %(error_type)s,
     %(attempts)s, %(latency_ms)s, %(input_tokens)s, %(output_tokens)s, %(cost)s)
"""


@dataclass
class CallRecord:
    """一次逻辑调用的统计记录(每次逻辑调用一行,attempts 记录重试次数)。"""

    call_type: str  # chat | chat_structured | chat_stream | embedding | rerank
    model: str
    status: str  # success | failed | rejected
    attempts: int
    latency_ms: int
    session_id: str | None = None
    error_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class UsageRecorder:
    """逐次调用统计,fire-and-forget 异步写 PG。绝不阻塞调用链路、绝不抛异常。"""

    def __init__(self, pool, source: str, pricing: dict[str, dict[str, float]]) -> None:
        self._pool = pool
        self._source = source  # api | worker | eval
        self._pricing = pricing

    def record(self, rec: CallRecord) -> None:
        try:
            asyncio.get_running_loop().create_task(self._write(rec))
        except Exception:  # noqa: BLE001 - 无事件循环等边界情况,只丢这一条记录
            logger.warning("llm_call_log 记录跳过", exc_info=True)

    async def _write(self, rec: CallRecord) -> None:
        try:
            cost = compute_cost(
                self._pricing, rec.model, rec.input_tokens, rec.output_tokens
            )
            async with get_cursor(self._pool) as cur:
                await cur.execute(_INSERT_SQL, {
                    "call_type": rec.call_type,
                    "model": rec.model,
                    "source": self._source,
                    "session_id": rec.session_id,
                    "status": rec.status,
                    "error_type": rec.error_type,
                    "attempts": rec.attempts,
                    "latency_ms": rec.latency_ms,
                    "input_tokens": rec.input_tokens,
                    "output_tokens": rec.output_tokens,
                    "cost": cost,
                })
        except Exception:  # noqa: BLE001 - 统计写失败绝不影响业务
            logger.warning("llm_call_log 写入失败", exc_info=True)
