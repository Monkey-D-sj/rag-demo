# LLM 调用治理层设计(限流/熔断、超时降级、成本统计、压测)

日期:2026-07-17
状态:已评审(设计经用户逐节确认)

## 1. 背景与目标

当前 LLM 调用层(`NormalModel` / `EmbeddingModel` / `QwenReranker`)只有 tenacity 重试,
没有限流、熔断、显式超时(chat)与成本统计。429 被映射为 `LLMRateLimitError` 且注释
"应由调用方限速而非无脑重试"——本设计补上这个缺口。

**目标:**

1. 限流:按调用类型(chat / embedding / rerank)的 RPM + 并发数双限,多实例共享配额(Redis)
2. 熔断:供应商持续故障时快速失败,自动探针恢复
3. 超时:chat 链路补显式超时;既有写死超时提为配置
4. 降级:快速失败 + SSE 友好报错;不做内容降级、不做备用模型
5. 成本统计:每次调用逐行入 PG,聚合 API + 前端统计页
6. 压测:Locust + mock LLM 替身,四场景,markdown 报告入 docs/

**非目标(明确砍掉/不做):**

- 缓存(embedding 缓存、语义缓存均不做)
- 备用模型切换、检索直出等内容降级
- TPM(token 每分钟)维度限流
- 流式中途重试(维持现状,留待后续)

## 2. 需求决策记录

| 决策点 | 结论 |
|---|---|
| 场景 | 多实例生产,限流/熔断状态放 Redis,全局共享配额 |
| 覆盖范围 | NormalModel + EmbeddingModel + QwenReranker 全部纳入 |
| 限流维度 | RPM + 并发数,按调用类型区分 |
| 降级策略 | 快速失败 + 友好报错(SSE error 事件) |
| 缓存 | 不做 |
| 成本统计 | PG 逐次记录 + 聚合 API + 前端统计页,价格表放配置 |
| 压测 | Locust + mock LLM(不烧真实 token),报告入 docs/loadtest/ |
| 架构方案 | 方案 A:Guard 组件 + 调用点组合(重试在外、guard 在每次尝试内) |

## 3. 总体架构

### 3.1 模块布局

新包 `rag/governance/`,与 `rag/observability/` 对等(关注点独立、独立配置类):

```
rag/governance/
├── __init__.py      # 导出 LLMGuard、GovernanceSettings
├── config.py        # GovernanceSettings(独立 BaseSettings,同 LangfuseSettings 模式)
├── limiter.py       # RedisRateLimiter —— RPM 滑动窗口 + 并发 ZSET 信号量 + 429 冷却
├── breaker.py       # RedisCircuitBreaker —— closed/open/half-open,状态存 Redis,Lua 原子迁移
├── guard.py         # LLMGuard —— 组合 limiter + breaker,提供 acquire() 上下文管理器与结果上报
├── usage.py         # UsageRecorder —— 逐次调用记录,asyncio.create_task 异步写 PG
└── pricing.py       # 价格表(配置驱动)→ 成本折算,纯函数
```

### 3.2 注入方式

三个模型类构造函数增加可选参数 `guard: LLMGuard | None = None`:

- `None`(默认)= 完全直通、零开销 —— eval 脚本、既有测试不改一行(null-object 模式,同 Langfuse)
- API `lifespan` 与 Worker `on_startup` 各自创建 `LLMGuard`(依赖 redis client + pg pool),
  传给三个模型实例
- 开关 `GOVERNANCE_ENABLED` 默认 `false`(遵循项目 `*_ENABLED` 惯例),`.env.example` 示范开启

### 3.3 调用点织入

以 `NormalModel.ainvoke` 为例 —— **重试在外、guard 在每次尝试内**:

```python
async for attempt in AsyncRetrying(...):          # 现有重试,不动
    with attempt:
        async with self._guard.acquire("chat"):   # 限流+熔断+并发,进入前检查
            async with self._translate():
                rsp = await self._model.ainvoke(messages)
# guard 退出时:成功/失败上报熔断器;逻辑调用结束后 usage/耗时异步写 PG
```

理由:熔断已打开时重试不应再打到供应商;每次尝试的失败逐次计入熔断器,语义最细。

`astream` 的织入:acquire 在流开始前,释放在 async generator 的 finally 中;
中途异常同样计入熔断器。

### 3.4 容错总原则:fail-open

Redis 不可用 → 记 warning、放行调用(可用性优先于限流严格性);统计写 PG 失败 → 只记日志。
治理层自身故障绝不能成为第二个故障源。

## 4. 核心机制

### 4.1 限流器 RedisRateLimiter(按 chat / embedding / rerank 各一份配额)

- **RPM(滑动窗口近似):** 相邻两个分钟桶加权(`gov:rpm:{type}:{minute}`,INCR + EXPIRE 120s,
  `count = 当前桶 + 上一桶 × 剩余占比`),Lua 脚本保证原子。超限时有界等待
  (带抖动轮询,默认最多 `ACQUIRE_MAX_WAIT_SECONDS`=10s),等不到 → 拒绝。
- **并发数(ZSET 信号量):** `ZADD gov:conc:{type} <now> <uuid>` 占坑,finally 中 `ZREM` 释放;
  计数前先 `ZREMRANGEBYSCORE` 清掉超过 120s 的僵尸成员 —— 实例崩溃未释放的坑位自动过期。
- **429 冷却:** 供应商返回 429 时写 `gov:cooldown:{type}`(TTL 取 `Retry-After`,无则默认 10s);
  acquire 先查冷却键,冷却期内进入等待/拒绝。

### 4.2 熔断器 RedisCircuitBreaker(按调用类型)

状态存 `gov:cb:{type}`(Redis hash),迁移用 Lua 保证多实例原子:

```
closed --连续失败 ≥ BREAKER_FAILURE_THRESHOLD--> open --冷却 BREAKER_COOLDOWN_SECONDS--> half-open
half-open --探针成功--> closed(清零)
half-open --探针失败--> open(重计冷却)
```

- **计入熔断:** `LLMRetryableError`(5xx)、网络错误、超时。
- **不计入:** 4xx 客户端错误、429(走冷却机制)、结构化输出解析/校验失败(模型行为问题,非可用性问题)。
- **half-open 探针:** `SETNX` 抢占,全集群只放行 1 个试探请求。
- 熔断打开期间 acquire 立即抛 `CircuitOpenError`,**不等待**。

### 4.3 超时

- `NormalModel` 的 `ChatOpenAI` 补显式 `timeout=LLM_TIMEOUT_SECONDS`(默认 60;
  流式场景为逐 chunk 读超时,不会误杀长回答);非流式调用外加 `asyncio.timeout` 兜底。
- `EmbeddingModel`(现写死 30s)、`QwenReranker`(现写死 30s)提为配置项。
- 超时映射为 `LLMTimeoutError(LLMRetryableError)` —— 可重试、计入熔断。

### 4.4 降级:快速失败 + 友好报错

新增异常挂进 `rag/common/exception.py` 现有体系,重试语义随基类自动生效
(前两个不可重试、立即停止;超时可重试):

| 异常 | 触发 | 用户看到的 SSE error 文案 |
|---|---|---|
| `RateLimitExceededError(LLMNonRetryableError)` | 本地限流等待超限 / 429 冷却中 | 当前咨询人数较多,请稍后重试 |
| `CircuitOpenError(LLMNonRetryableError)` | 熔断打开 | AI 服务暂时不可用,请稍后重试 |
| `LLMTimeoutError(LLMRetryableError)` | 超时(重试用尽后) | 回答生成超时,请重试 |

- **chat SSE 链路:** producer 现为 `stream.error(str(e))`(原始异常直达用户),改为经
  `friendly_message(exc)` 映射;治理异常给上表文案,其余给通用文案;原始异常完整进日志。
- **Worker 实体抽取:** 治理拒绝按普通失败传播,由既有 `graph_status=failed` + cron
  指数退避自愈接住,零改动。
- **检索链路(已存在,验证过,不动):** 向量腿失败 → 纯 BM25;rerank 失败 → 原序透传。

### 4.5 新增配置(GovernanceSettings 默认值)

```
GOVERNANCE_ENABLED=false
CHAT_RPM_LIMIT=60          CHAT_MAX_CONCURRENCY=8
EMBEDDING_RPM_LIMIT=500    EMBEDDING_MAX_CONCURRENCY=10
RERANK_RPM_LIMIT=120       RERANK_MAX_CONCURRENCY=8
ACQUIRE_MAX_WAIT_SECONDS=10
BREAKER_FAILURE_THRESHOLD=5
BREAKER_COOLDOWN_SECONDS=30
LLM_TIMEOUT_SECONDS=60
EMBEDDING_TIMEOUT_SECONDS=30
RERANK_TIMEOUT_SECONDS=30
COOLDOWN_DEFAULT_SECONDS=10
LLM_PRICING={}   # JSON:{"模型名": {"input": 每百万token价, "output": ...}}
```

## 5. 成本统计

### 5.1 PG 表 llm_call_log(Alembic 迁移)

```sql
id            BIGSERIAL PRIMARY KEY
created_at    timestamptz NOT NULL DEFAULT now()
call_type     text NOT NULL      -- chat | chat_structured | chat_stream | embedding | rerank
model         text NOT NULL
source        text NOT NULL      -- api | worker | eval
session_id    text               -- chat 调用带,其余 NULL
status        text NOT NULL      -- success | failed | rejected(被限流/熔断拒绝)
error_type    text               -- 异常类名,成功为 NULL
attempts      int NOT NULL       -- 实际尝试次数(含重试)
latency_ms    int NOT NULL
input_tokens  int                -- SDK usage 字段;拿不到为 NULL
output_tokens int
cost          numeric(12, 6)     -- 按价格表折算(元);模型不在价格表则 NULL
-- 索引:(created_at)、(model, created_at)
```

### 5.2 写入路径

- 记录在**逻辑调用结束时**组装(重试循环之外;guard 每次尝试退出只上报熔断器),
  `asyncio.create_task` 发射后不管,内部 try/except 全兜底。
- **每次逻辑调用一行**(非每次重试一行),`attempts` 记录重试次数。
- **被拒绝的调用也记录**(`status=rejected`,tokens 为 NULL)—— 压测验证限流生效的依据。
- Token 来源:非流式取 `AIMessage.usage_metadata` / `rsp.usage`;流式开
  `stream_usage=True`,末尾 chunk 携带 usage;rerank 只记次数与耗时。

### 5.3 聚合 API(新模块 rag/api/modules/stats/)

- `GET /api/stats/llm/summary?start=&end=&group_by=day|model|source`
  → 调用数、成功/失败/拒绝数、token 合计、成本合计、平均/P95 延迟(`percentile_cont`)
- `GET /api/stats/llm/recent?limit=50` → 最近调用明细

### 5.4 前端统计页

新增 `StatsPage`(路由与 Chat/Documents/Health 并列):按天成本/调用量趋势图、
按模型分布表、成功率。图表实现遵循 dataviz 技能规范。

## 6. 负载压测

`loadtest/` 目录(不进 rag 包):

```
loadtest/
├── mock_llm.py      # OpenAI 兼容 mock:/chat/completions(含流式)、/embeddings、/v1/reranks
│                    #   环境变量控制:MOCK_LATENCY_MS、MOCK_ERROR_RATE、MOCK_ERROR_CODE
├── locustfile.py    # 压 /chat SSE 端点(完整读流),西游记主题问题池
└── README.md        # 跑法说明
```

四场景,报告落 `docs/loadtest/YYYY-MM-DD-report.md`:

| 场景 | 设置 | 验证目标 |
|---|---|---|
| S1 基线 | mock 正常,低并发 | 延迟分位数基线(P50/P95/P99)、吞吐 |
| S2 限流饱和 | 并发超过 RPM×并发配额 | 超额请求收到友好拒绝而非雪崩;rejected 计数与配额吻合 |
| S3 供应商故障 | mock 100% 返 500,中途恢复 | 熔断 ≤5 次失败后打开、期间快速失败、恢复后探针自动闭合 |
| S4 慢供应商 | mock 延迟超过超时阈值 | 超时生效、不堆积连接、并发坑位正常释放 |

报告内容:各场景 RPS / 延迟分位数 / 错误率、熔断状态迁移时间线(日志提取)、
`llm_call_log` 统计对账、结论与配额调参建议。

## 7. 测试策略

- **单元测试(默认跑):** limiter/breaker 用 `fakeredis`(新 dev 依赖)测窗口滚动、
  并发泄漏回收、三态迁移、fail-open;guard 用假 limiter/breaker 测织入顺序;
  pricing、`friendly_message` 纯函数直测;`tests/test_normal.py` 增加 guard=None 直通回归。
- **集成测试(`-m integration`):** 真 Redis 下多协程并发抢配额;`llm_call_log`
  写入与聚合 API 端到端。
- **压测(手动):** 第 6 节,不进 CI。

## 8. 实施边界

- 改动的既有文件:`rag/models/normal.py`、`rag/models/embedding.py`、`rag/models/rerank.py`
  (各织入 guard,约 5-10 行/处)、`rag/common/exception.py`(新异常)、
  `rag/api/main.py` / `rag/worker/main.py`(创建并注入 guard)、chat producer(友好报错映射)、
  `.env.example`、`pyproject.toml`(locust、fakeredis 为 dev 依赖)
- 新增:`rag/governance/`、`rag/api/modules/stats/`、`frontend` StatsPage、`loadtest/`、
  Alembic 迁移一份
- 不动:检索降级路径、Worker 状态机、eval 模块
