# 负载压测

LLM 用本地 mock 替身(不烧真实 token),验证的是**自身治理层行为**:
限流是否生效、熔断是否按时打开/恢复、超时是否兜底、降级文案是否返回。

## 准备

1. 起基础设施与 mock:

```bash
docker compose up -d postgres redis minio
uv run alembic upgrade head
uv run uvicorn loadtest.mock_llm:app --port 9100   # 终端 A
```

2. `.env` 里把模型端点指向 mock(压测后记得改回):

```bash
MODEL_URL=http://localhost:9100/v1
EMBEDDING_URL=http://localhost:9100/v1
GOVERNANCE_ENABLED=true
```

3. 起 API:`uv run rag-api`(终端 B)

## 四场景

| 场景 | mock 环境变量 | locust 参数 |
|---|---|---|
| S1 基线 | `MOCK_LATENCY_MS=200` | `-u 10 -r 2 -t 2m` |
| S2 限流饱和 | `MOCK_LATENCY_MS=200` | `-u 60 -r 10 -t 2m`(超过 CHAT_RPM_LIMIT=60 × 并发 8) |
| S3 供应商故障 | 前半程 `MOCK_ERROR_RATE=1 MOCK_ERROR_CODE=500`,1 分钟后改回 0 重启 mock | `-u 10 -r 2 -t 3m` |
| S4 慢供应商 | `MOCK_LATENCY_MS=70000`(> LLM_TIMEOUT_SECONDS=60) | `-u 5 -r 1 -t 2m` |

跑法(每场景):

```bash
uv run locust -f loadtest/locustfile.py --headless -u <U> -r <R> -t <T> \
  --host http://localhost:8000 --csv loadtest/results/<场景名>
```

## 对账

压测后从 llm_call_log 取治理层视角的数据:

```sql
SELECT status, count(*), round(avg(latency_ms)) AS avg_ms
FROM llm_call_log WHERE created_at > now() - interval '10 minutes'
GROUP BY status;
```

熔断状态迁移时间线:`grep "熔断器打开" API 日志`。

报告写入 `docs/loadtest/YYYY-MM-DD-report.md`(模板见实施计划 Task 13)。
