# LLM 调用治理层实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 NormalModel / EmbeddingModel / QwenReranker 三个模型调用点织入 Redis 共享的限流/熔断/超时/降级治理层,逐次调用成本入 PG 并提供聚合 API 与前端统计页,最后用 Locust + mock LLM 完成四场景压测报告。

**Architecture:** 新包 `rag/governance/`(LLMGuard = 限流器 + 熔断器 + 统计记录器的组合),模型类构造函数加可选 `guard` 参数(None = 零开销直通,null-object 模式)。织入顺序:重试在外、guard 在每次尝试内。容错总原则 fail-open:Redis/PG 故障绝不阻断业务调用。

**Tech Stack:** Python 3.12+ / redis.asyncio(Lua 脚本)/ psycopg3 / FastAPI / Alembic / fakeredis[lua](单测)/ Locust + FastAPI mock(压测)/ React 18 + TailwindCSS(统计页)

**Spec:** `docs/superpowers/specs/2026-07-17-llm-governance-design.md`(已批准,实现遇到歧义以 spec 为准)

## Global Constraints

- Python `>=3.12`(`asyncio.timeout` 可用);依赖管理用 `uv add` / `uv add --dev`,禁止手改 lock
- 日志一律 `from rag.common.logging import get_logger`,禁止 `logging.getLogger(__name__)`
- 禁止重复声明已有的变量(项目 CLAUDE.md 规范)
- pytest `asyncio_mode=auto`:测试函数直接 `async def`,不加 `@pytest.mark.asyncio`
- 默认测试命令跳过 integration/eval marker:`uv run pytest tests/ -v --ignore=tests/test_db.py --ignore=tests/test_db_neo4j.py`
- fail-open:治理层内所有 Redis/PG 操作必须 try/except 兜底,失败只记 warning、放行业务
- 新配置全部进 `GovernanceSettings`(独立 BaseSettings),不动 `rag/config.py` 的 `Settings`
- Redis key 统一 `gov:` 前缀;配额桶(quota)取值 `chat` / `embedding` / `rerank`
- 统计 call_type 取值:`chat` / `chat_structured` / `chat_stream` / `embedding` / `rerank`;status 取值:`success` / `failed` / `rejected`
- 提交信息结尾:`Co-Authored-By: Claude <noreply@anthropic.com>`

---

### Task 1: 治理异常与友好文案(rag/common/exception.py)

**Files:**
- Modify: `rag/common/exception.py`(在第 69 行 `LLMRateLimitError` 类之后、`# ── 错误码映射` 之前插入)
- Test: `tests/test_exception_governance.py`(新建)

**Interfaces:**
- Produces:
  - `LLMTimeoutError(LLMRetryableError)` — 构造同基类 `(message, status_code=0, model="")`
  - `GovernanceRejectedError(LLMNonRetryableError)` — 治理拒绝基类
  - `RateLimitExceededError(GovernanceRejectedError)`
  - `CircuitOpenError(GovernanceRejectedError)`
  - `friendly_message(exc: BaseException) -> str` — 任意异常 → 用户可见文案

- [ ] **Step 1: Write the failing test**

创建 `tests/test_exception_governance.py`:

```python
from rag.common.exception import (
    CircuitOpenError,
    LLMTimeoutError,
    RateLimitExceededError,
    friendly_message,
    is_retryable,
)


def test_timeout_is_retryable():
    assert is_retryable(LLMTimeoutError("超时")) is True


def test_governance_rejections_are_non_retryable():
    assert is_retryable(RateLimitExceededError("限流")) is False
    assert is_retryable(CircuitOpenError("熔断")) is False


def test_friendly_message_mapping():
    assert friendly_message(RateLimitExceededError("x")) == "当前咨询人数较多,请稍后重试"
    assert friendly_message(CircuitOpenError("x")) == "AI 服务暂时不可用,请稍后重试"
    assert friendly_message(LLMTimeoutError("x")) == "回答生成超时,请重试"


def test_friendly_message_fallback_hides_raw_error():
    msg = friendly_message(RuntimeError("Traceback: secret internal detail"))
    assert msg == "服务开小差了,请稍后重试"
    assert "secret" not in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exception_governance.py -v`
Expected: FAIL — `ImportError: cannot import name 'LLMTimeoutError'`

- [ ] **Step 3: Write minimal implementation**

在 `rag/common/exception.py` 的 `LLMRateLimitError` 类定义之后插入:

```python
class LLMTimeoutError(LLMRetryableError):
    """请求超时 — 可重试,计入熔断"""
    pass


# ── 治理层拒绝（本地产生，非供应商响应） ──────────────────

class GovernanceRejectedError(LLMNonRetryableError):
    """治理层拒绝基类:限流/熔断在本地拦截,未打到供应商"""
    pass


class RateLimitExceededError(GovernanceRejectedError):
    """本地限流:RPM/并发等待超限或 429 冷却中"""
    pass


class CircuitOpenError(GovernanceRejectedError):
    """熔断打开,快速失败"""
    pass


# ── 用户可见的友好文案（chat SSE error 事件用） ────────────

_FRIENDLY_MESSAGES: list[tuple[type[BaseException], str]] = [
    (RateLimitExceededError, "当前咨询人数较多,请稍后重试"),
    (CircuitOpenError, "AI 服务暂时不可用,请稍后重试"),
    (LLMTimeoutError, "回答生成超时,请重试"),
]

_DEFAULT_FRIENDLY = "服务开小差了,请稍后重试"


def friendly_message(exc: BaseException) -> str:
    """异常 → 用户可见文案。原始异常信息绝不外泄,只进日志。"""
    for exc_type, msg in _FRIENDLY_MESSAGES:
        if isinstance(exc, exc_type):
            return msg
    return _DEFAULT_FRIENDLY
```

- [ ] **Step 4: Run tests to verify pass(含既有回归)**

Run: `uv run pytest tests/test_exception_governance.py tests/test_normal.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/common/exception.py tests/test_exception_governance.py
git commit -m "feat(governance): 治理层异常类型与友好文案映射"
```

---

### Task 2: GovernanceSettings 与价格折算(rag/governance/config.py, pricing.py)

**Files:**
- Create: `rag/governance/__init__.py`(本任务先留空,Task 6 填充)
- Create: `rag/governance/config.py`
- Create: `rag/governance/pricing.py`
- Test: `tests/test_governance_config.py`(新建)

**Interfaces:**
- Produces:
  - `GovernanceSettings` — 全部治理配置字段(默认值见代码);方法 `limits_for(quota: str) -> tuple[int, int]`(返回 `(rpm_limit, max_concurrency)`)、`pricing() -> dict[str, dict[str, float]]`
  - `get_governance_settings() -> GovernanceSettings`(lru_cache 单例)
  - `compute_cost(pricing: dict[str, dict[str, float]], model: str, input_tokens: int | None, output_tokens: int | None) -> float | None`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_governance_config.py`:

```python
from rag.governance.config import GovernanceSettings
from rag.governance.pricing import compute_cost


def _settings(**kw) -> GovernanceSettings:
    # _env_file=None:隔离本机 .env,只用默认值+显式覆盖
    return GovernanceSettings(_env_file=None, **kw)


def test_defaults_disabled_and_limits():
    s = _settings()
    assert s.GOVERNANCE_ENABLED is False
    assert s.limits_for("chat") == (60, 8)
    assert s.limits_for("embedding") == (500, 10)
    assert s.limits_for("rerank") == (120, 8)


def test_pricing_parses_json():
    s = _settings(LLM_PRICING='{"deepseek-chat": {"input": 2.0, "output": 8.0}}')
    assert s.pricing() == {"deepseek-chat": {"input": 2.0, "output": 8.0}}


def test_pricing_invalid_json_falls_back_to_empty():
    s = _settings(LLM_PRICING="not-json")
    assert s.pricing() == {}


def test_compute_cost_per_million_tokens():
    pricing = {"m": {"input": 2.0, "output": 8.0}}
    # 1000 输入 + 500 输出 → 2*1000/1e6 + 8*500/1e6 = 0.006
    assert compute_cost(pricing, "m", 1000, 500) == 0.006


def test_compute_cost_unknown_model_or_no_tokens_is_none():
    assert compute_cost({}, "unknown", 100, 100) is None
    assert compute_cost({"m": {"input": 1, "output": 1}}, "m", None, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_governance_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.governance'`

- [ ] **Step 3: Write implementation**

创建 `rag/governance/__init__.py`(空文件即可,Task 6 填充导出)。

创建 `rag/governance/config.py`:

```python
import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.common.logging import get_logger

logger = get_logger()


class GovernanceSettings(BaseSettings):
    """LLM 调用治理配置。

    不并入 rag.config.Settings —— 治理关注点独立成模块(同 LangfuseSettings 模式)。
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GOVERNANCE_ENABLED: bool = False

    # ── 限流(按配额桶) ──
    CHAT_RPM_LIMIT: int = 60
    CHAT_MAX_CONCURRENCY: int = 8
    EMBEDDING_RPM_LIMIT: int = 500
    EMBEDDING_MAX_CONCURRENCY: int = 10
    RERANK_RPM_LIMIT: int = 120
    RERANK_MAX_CONCURRENCY: int = 8
    ACQUIRE_MAX_WAIT_SECONDS: float = 10.0
    COOLDOWN_DEFAULT_SECONDS: float = 10.0

    # ── 熔断 ──
    BREAKER_FAILURE_THRESHOLD: int = 5
    BREAKER_COOLDOWN_SECONDS: float = 30.0

    # ── 超时 ──
    LLM_TIMEOUT_SECONDS: float = 60.0
    EMBEDDING_TIMEOUT_SECONDS: float = 30.0
    RERANK_TIMEOUT_SECONDS: float = 30.0

    # ── 价格表:JSON,{"模型名": {"input": 每百万token价, "output": ...}} ──
    LLM_PRICING: str = "{}"

    def limits_for(self, quota: str) -> tuple[int, int]:
        """配额桶 → (rpm_limit, max_concurrency)。"""
        return {
            "chat": (self.CHAT_RPM_LIMIT, self.CHAT_MAX_CONCURRENCY),
            "embedding": (self.EMBEDDING_RPM_LIMIT, self.EMBEDDING_MAX_CONCURRENCY),
            "rerank": (self.RERANK_RPM_LIMIT, self.RERANK_MAX_CONCURRENCY),
        }[quota]

    def pricing(self) -> dict[str, dict[str, float]]:
        """解析价格表 JSON;解析失败降级为空表(成本记 NULL),不阻断启动。"""
        try:
            parsed = json.loads(self.LLM_PRICING)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        logger.warning("LLM_PRICING 不是合法 JSON,成本折算将全部记 NULL")
        return {}


@lru_cache
def get_governance_settings() -> GovernanceSettings:
    return GovernanceSettings()
```

创建 `rag/governance/pricing.py`:

```python
def compute_cost(
    pricing: dict[str, dict[str, float]],
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """按每百万 token 单价折算成本(元)。模型不在价格表或无 token 数据 → None。"""
    price = pricing.get(model)
    if price is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    total = (input_tokens or 0) * price.get("input", 0.0) + (
        output_tokens or 0
    ) * price.get("output", 0.0)
    return round(total / 1_000_000, 6)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_governance_config.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/governance/ tests/test_governance_config.py
git commit -m "feat(governance): GovernanceSettings 配置与价格折算"
```

---

### Task 3: 限流器 RedisRateLimiter(rag/governance/limiter.py)

**Files:**
- Modify: `pyproject.toml`(经 `uv add --dev "fakeredis[lua]"`)
- Create: `rag/governance/limiter.py`
- Test: `tests/test_governance_limiter.py`(新建)

**Interfaces:**
- Consumes: `GovernanceSettings.limits_for/ACQUIRE_MAX_WAIT_SECONDS/COOLDOWN_DEFAULT_SECONDS`(Task 2)、`RateLimitExceededError`(Task 1)
- Produces: `RedisRateLimiter(redis, settings)`:
  - `async acquire(quota: str) -> str | None` — 通过返回并发坑位 member(uuid hex);Redis 故障 fail-open 返回 None;等待超限抛 `RateLimitExceededError`
  - `async release(quota: str, member: str | None) -> None`
  - `async start_cooldown(quota: str, seconds: float | None = None) -> None`
  - 实例属性 `_now`(默认 `time.time`,测试可注入假时钟)

- [ ] **Step 1: Add dev dependency**

Run: `uv add --dev "fakeredis[lua]"`
Expected: pyproject `[dependency-groups].dev` 出现 `fakeredis[lua]>=...`,lock 更新

- [ ] **Step 2: Write the failing test**

创建 `tests/test_governance_limiter.py`:

```python
import pytest
from fakeredis.aioredis import FakeRedis

from rag.common.exception import RateLimitExceededError
from rag.governance.config import GovernanceSettings
from rag.governance.limiter import RedisRateLimiter


def _limiter(redis, **kw) -> RedisRateLimiter:
    settings = GovernanceSettings(
        _env_file=None, ACQUIRE_MAX_WAIT_SECONDS=0, **kw
    )  # 等待窗口 0:超限立即拒绝,测试不耗时
    return RedisRateLimiter(redis, settings)


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


async def test_acquire_release_roundtrip(redis):
    lim = _limiter(redis)
    member = await lim.acquire("chat")
    assert member is not None
    assert await redis.zcard("gov:conc:chat") == 1
    await lim.release("chat", member)
    assert await redis.zcard("gov:conc:chat") == 0


async def test_concurrency_limit_rejects(redis):
    lim = _limiter(redis, CHAT_MAX_CONCURRENCY=1)
    await lim.acquire("chat")
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")


async def test_stale_concurrency_slots_are_evicted(redis):
    lim = _limiter(redis, CHAT_MAX_CONCURRENCY=1)
    t = 1000.0
    lim._now = lambda: t
    await lim.acquire("chat")  # 占坑但不释放(模拟实例崩溃)
    t += 121  # 超过 120s 僵尸阈值
    member = await lim.acquire("chat")  # 僵尸被清,坑位可复用
    assert member is not None


async def test_rpm_limit_rejects(redis):
    lim = _limiter(redis, CHAT_RPM_LIMIT=2)
    lim._now = lambda: 1000.0  # 固定时钟:避免跨分钟桶导致的窗口加权波动
    await lim.release("chat", await lim.acquire("chat"))
    await lim.release("chat", await lim.acquire("chat"))
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")


async def test_rpm_rejection_does_not_leak_concurrency_slot(redis):
    lim = _limiter(redis, CHAT_RPM_LIMIT=1)
    lim._now = lambda: 1000.0
    await lim.release("chat", await lim.acquire("chat"))
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")
    assert await redis.zcard("gov:conc:chat") == 0


async def test_cooldown_blocks_acquire(redis):
    lim = _limiter(redis)
    await lim.start_cooldown("chat")
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")


async def test_redis_failure_fails_open():
    class _BrokenRedis:
        def __getattr__(self, name):
            async def _fail(*a, **kw):
                raise ConnectionError("redis down")
            return _fail

    lim = _limiter(_BrokenRedis())
    assert await lim.acquire("chat") is None  # fail-open 放行
    await lim.release("chat", None)  # 不抛
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_governance_limiter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.governance.limiter'`

- [ ] **Step 4: Write implementation**

创建 `rag/governance/limiter.py`:

```python
import asyncio
import random
import time
import uuid

from rag.common.exception import RateLimitExceededError
from rag.common.logging import get_logger
from rag.governance.config import GovernanceSettings

logger = get_logger()

# 相邻两分钟桶加权近似滑动窗口。
# KEYS[1]=当前分钟桶 KEYS[2]=上一分钟桶
# ARGV[1]=limit ARGV[2]=当前分钟内已经过秒数(0-59.999)
# 返回 1=放行(已计数) 0=超限(未计数)
_RPM_LUA = """
local cur = tonumber(redis.call('GET', KEYS[1]) or '0')
local prev = tonumber(redis.call('GET', KEYS[2]) or '0')
local weight = 1 - (tonumber(ARGV[2]) / 60)
if cur + prev * weight >= tonumber(ARGV[1]) then
    return 0
end
local n = redis.call('INCR', KEYS[1])
if n == 1 then
    redis.call('EXPIRE', KEYS[1], 120)
end
return 1
"""

# 并发坑位超过此秒数视为僵尸(实例崩溃未释放),计数前清除
_STALE_SLOT_SECONDS = 120


class RedisRateLimiter:
    """RPM 滑动窗口 + 并发 ZSET 信号量 + 429 冷却,状态全在 Redis,多实例共享。

    fail-open:Redis 故障时记 warning 放行(acquire 返回 None,release 跳过)。
    """

    def __init__(self, redis, settings: GovernanceSettings) -> None:
        self._redis = redis
        self._settings = settings
        self._now = time.time  # 可注入假时钟,供测试控制窗口滚动

    async def acquire(self, quota: str) -> str | None:
        """获取配额,返回并发坑位 member(release 时传回)。

        超限时有界等待(带抖动轮询),到 ACQUIRE_MAX_WAIT_SECONDS 仍拿不到
        则抛 RateLimitExceededError。
        """
        rpm_limit, max_conc = self._settings.limits_for(quota)
        deadline = time.monotonic() + self._settings.ACQUIRE_MAX_WAIT_SECONDS
        while True:
            try:
                ok, detail = await self._try_acquire(quota, rpm_limit, max_conc)
            except Exception:  # noqa: BLE001 - fail-open:限流器故障不阻断业务
                logger.warning("限流器 Redis 异常,fail-open 放行: %s", quota, exc_info=True)
                return None
            if ok:
                return detail
            if time.monotonic() >= deadline:
                raise RateLimitExceededError(f"本地限流({quota}): {detail}", status_code=429)
            await asyncio.sleep(0.2 + random.random() * 0.3)

    async def _try_acquire(
        self, quota: str, rpm_limit: int, max_conc: int
    ) -> tuple[bool, str]:
        """单次尝试。返回 (True, member) 或 (False, 拒因)。"""
        # 1. 429 冷却检查
        if await self._redis.exists(f"gov:cooldown:{quota}"):
            return False, "429 冷却中"
        # 2. 并发坑位:先清僵尸,ZADD 占坑后查总数,超了回滚自己
        now = self._now()
        conc_key = f"gov:conc:{quota}"
        member = uuid.uuid4().hex
        await self._redis.zremrangebyscore(conc_key, 0, now - _STALE_SLOT_SECONDS)
        await self._redis.zadd(conc_key, {member: now})
        if await self._redis.zcard(conc_key) > max_conc:
            await self._redis.zrem(conc_key, member)
            return False, "并发已满"
        # 3. RPM 滑动窗口(Lua 原子判定+计数)
        minute = int(now // 60)
        allowed = await self._redis.eval(
            _RPM_LUA,
            2,
            f"gov:rpm:{quota}:{minute}",
            f"gov:rpm:{quota}:{minute - 1}",
            rpm_limit,
            now % 60,
        )
        if not int(allowed):
            await self._redis.zrem(conc_key, member)
            return False, "RPM 超限"
        return True, member

    async def release(self, quota: str, member: str | None) -> None:
        """释放并发坑位。member=None(fail-open 时)跳过。"""
        if member is None:
            return
        try:
            await self._redis.zrem(f"gov:conc:{quota}", member)
        except Exception:  # noqa: BLE001 - 释放失败由僵尸清理兜底
            logger.warning("释放并发坑位失败(僵尸清理会兜底): %s", quota, exc_info=True)

    async def start_cooldown(self, quota: str, seconds: float | None = None) -> None:
        """供应商 429 后开启冷却:所有实例在 TTL 内拒绝发起该桶的调用。"""
        ttl = seconds if seconds and seconds > 0 else self._settings.COOLDOWN_DEFAULT_SECONDS
        try:
            await self._redis.set(f"gov:cooldown:{quota}", "1", ex=max(1, int(ttl)))
        except Exception:  # noqa: BLE001
            logger.warning("写入 429 冷却键失败: %s", quota, exc_info=True)
```

- [ ] **Step 5: Run test to verify pass**

Run: `uv run pytest tests/test_governance_limiter.py -v`
Expected: 7 PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock rag/governance/limiter.py tests/test_governance_limiter.py
git commit -m "feat(governance): Redis 限流器 - RPM 滑动窗口 + 并发信号量 + 429 冷却"
```

---

### Task 4: 熔断器 RedisCircuitBreaker(rag/governance/breaker.py)

**Files:**
- Create: `rag/governance/breaker.py`
- Test: `tests/test_governance_breaker.py`(新建)

**Interfaces:**
- Consumes: `GovernanceSettings.BREAKER_FAILURE_THRESHOLD/BREAKER_COOLDOWN_SECONDS`(Task 2)、`CircuitOpenError`(Task 1)
- Produces: `RedisCircuitBreaker(redis, settings)`:
  - `async check(quota: str) -> str | None` — 正常放行返回 None;half-open 抢到探针返回 `"probe"`;熔断打开/探针被占抛 `CircuitOpenError`;Redis 故障 fail-open 返回 None
  - `async record_success(quota: str, probe: str | None) -> None`
  - `async record_failure(quota: str, probe: str | None) -> None`
  - `async release_probe(quota: str) -> None` — 探针请求未真正发出(如被限流拒绝)时归还探针
  - 实例属性 `_now`(默认 `time.time`,测试注入)

**原子性说明(spec 4.2 的落地取舍):** 探针互斥用 `SET NX`(原子);open 状态写入是幂等操作,双实例同时触发写入结果相同,竞态无害 —— 因此不需要整段 Lua,代码更简单可测。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_governance_breaker.py`:

```python
import pytest
from fakeredis.aioredis import FakeRedis

from rag.common.exception import CircuitOpenError
from rag.governance.breaker import RedisCircuitBreaker
from rag.governance.config import GovernanceSettings


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


def _breaker(redis, **kw) -> RedisCircuitBreaker:
    settings = GovernanceSettings(
        _env_file=None, BREAKER_FAILURE_THRESHOLD=3, BREAKER_COOLDOWN_SECONDS=30, **kw
    )
    return RedisCircuitBreaker(redis, settings)


async def test_closed_allows(redis):
    b = _breaker(redis)
    assert await b.check("chat") is None


async def test_opens_after_consecutive_failures(redis):
    b = _breaker(redis)
    for _ in range(3):
        await b.record_failure("chat", probe=None)
    with pytest.raises(CircuitOpenError):
        await b.check("chat")


async def test_success_resets_failure_streak(redis):
    b = _breaker(redis)
    await b.record_failure("chat", probe=None)
    await b.record_failure("chat", probe=None)
    await b.record_success("chat", probe=None)  # 连续中断,计数清零
    await b.record_failure("chat", probe=None)
    assert await b.check("chat") is None  # 未达阈值,仍闭合


async def test_half_open_single_probe_then_close(redis):
    b = _breaker(redis)
    t = 1000.0
    b._now = lambda: t
    for _ in range(3):
        await b.record_failure("chat", probe=None)
    t += 31  # 冷却期满 → half-open
    assert await b.check("chat") == "probe"  # 第一个请求拿到探针
    with pytest.raises(CircuitOpenError):
        await b.check("chat")  # 第二个请求被拒(探针已被占)
    await b.record_success("chat", probe="probe")  # 探针成功 → 闭合
    assert await b.check("chat") is None


async def test_probe_failure_reopens(redis):
    b = _breaker(redis)
    t = 1000.0
    b._now = lambda: t
    for _ in range(3):
        await b.record_failure("chat", probe=None)
    t += 31
    assert await b.check("chat") == "probe"
    await b.record_failure("chat", probe="probe")  # 探针失败 → 重新 open
    t += 10  # 未满新一轮冷却
    with pytest.raises(CircuitOpenError):
        await b.check("chat")


async def test_redis_failure_fails_open():
    class _BrokenRedis:
        def __getattr__(self, name):
            async def _fail(*a, **kw):
                raise ConnectionError("redis down")
            return _fail

    b = _breaker(_BrokenRedis())
    assert await b.check("chat") is None  # fail-open
    await b.record_failure("chat", probe=None)  # 不抛
    await b.record_success("chat", probe=None)  # 不抛
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_governance_breaker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.governance.breaker'`

- [ ] **Step 3: Write implementation**

创建 `rag/governance/breaker.py`:

```python
import time

from rag.common.exception import CircuitOpenError
from rag.common.logging import get_logger
from rag.governance.config import GovernanceSettings

logger = get_logger()

# 探针键 TTL:探针方崩溃时最多阻塞 half-open 这么多秒,之后其他实例可再探
_PROBE_TTL_SECONDS = 60


class RedisCircuitBreaker:
    """三态熔断器,状态存 Redis hash(gov:cb:{quota}),多实例共享。

    closed --连续失败≥阈值--> open --冷却期满--> half-open(SET NX 抢探针)
    half-open 探针成功 → closed(清零);探针失败 → open(重计冷却)。
    fail-open:Redis 故障时放行,状态操作失败只记日志。
    """

    def __init__(self, redis, settings: GovernanceSettings) -> None:
        self._redis = redis
        self._settings = settings
        self._now = time.time  # 可注入假时钟

    def _key(self, quota: str) -> str:
        return f"gov:cb:{quota}"

    async def check(self, quota: str) -> str | None:
        """入口检查。None=放行;"probe"=half-open 探针放行;打开则抛 CircuitOpenError。"""
        try:
            data = await self._redis.hgetall(self._key(quota))
        except Exception:  # noqa: BLE001 - fail-open
            logger.warning("熔断器 Redis 异常,fail-open 放行: %s", quota, exc_info=True)
            return None
        if not data or data.get("state") != "open":
            return None
        opened_at = float(data.get("opened_at", 0))
        if self._now() - opened_at < self._settings.BREAKER_COOLDOWN_SECONDS:
            raise CircuitOpenError(f"熔断打开({quota}),快速失败", status_code=503)
        # 冷却期满 → half-open:全集群只放行一个探针
        try:
            got = await self._redis.set(
                f"{self._key(quota)}:probe", "1", nx=True, ex=_PROBE_TTL_SECONDS
            )
        except Exception:  # noqa: BLE001
            logger.warning("抢占熔断探针失败,fail-open 放行: %s", quota, exc_info=True)
            return None
        if got:
            return "probe"
        raise CircuitOpenError(f"熔断半开({quota}),探针已被占用", status_code=503)

    async def record_success(self, quota: str, probe: str | None) -> None:
        try:
            if probe:
                # 探针成功 → 闭合:删状态与探针键
                await self._redis.delete(self._key(quota), f"{self._key(quota)}:probe")
            else:
                # 普通成功 → 连续失败清零
                await self._redis.hset(self._key(quota), "failures", 0)
        except Exception:  # noqa: BLE001
            logger.warning("熔断器记录成功失败: %s", quota, exc_info=True)

    async def record_failure(self, quota: str, probe: str | None) -> None:
        try:
            key = self._key(quota)
            if probe:
                # 探针失败 → 重新 open,重计冷却
                await self._redis.hset(
                    key, mapping={"state": "open", "opened_at": self._now(), "failures": 0}
                )
                await self._redis.delete(f"{key}:probe")
                return
            failures = await self._redis.hincrby(key, "failures", 1)
            if failures >= self._settings.BREAKER_FAILURE_THRESHOLD:
                await self._redis.hset(
                    key, mapping={"state": "open", "opened_at": self._now()}
                )
                logger.warning("熔断器打开: %s(连续失败 %d 次)", quota, failures)
        except Exception:  # noqa: BLE001
            logger.warning("熔断器记录失败异常: %s", quota, exc_info=True)

    async def release_probe(self, quota: str) -> None:
        """探针请求未真正发出(如被限流拒绝)时归还探针,让其他请求可探。"""
        try:
            await self._redis.delete(f"{self._key(quota)}:probe")
        except Exception:  # noqa: BLE001
            logger.warning("归还熔断探针失败: %s", quota, exc_info=True)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_governance_breaker.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/governance/breaker.py tests/test_governance_breaker.py
git commit -m "feat(governance): Redis 熔断器 - closed/open/half-open 三态与探针恢复"
```

---

### Task 5: 统计记录 UsageRecorder + llm_call_log 迁移

**Files:**
- Modify: `rag/common/logging.py`(在 `reset_session` 函数后添加 `current_session_id`)
- Create: `rag/governance/usage.py`
- Create: `alembic/versions/0010_llm_call_log.py`
- Test: `tests/test_governance_usage.py`(新建)

**Interfaces:**
- Consumes: `compute_cost`(Task 2)、`get_cursor`(`rag.db.postgres`,已有)
- Produces:
  - `current_session_id() -> str | None`(rag/common/logging.py)
  - `CallRecord` dataclass — 字段:`call_type: str, model: str, status: str, attempts: int, latency_ms: int, session_id: str | None = None, error_type: str | None = None, input_tokens: int | None = None, output_tokens: int | None = None`
  - `UsageRecorder(pool, source: str, pricing: dict[str, dict[str, float]])`:
    - `record(rec: CallRecord) -> None` — fire-and-forget,绝不抛异常/阻塞
    - `async _write(rec: CallRecord) -> None` — 内部,补 cost 后 INSERT

- [ ] **Step 1: Write the failing test**

创建 `tests/test_governance_usage.py`:

```python
import asyncio
from contextlib import asynccontextmanager

import rag.governance.usage as usage_mod
from rag.governance.usage import CallRecord, UsageRecorder


class _FakeCursor:
    def __init__(self):
        self.executed = []

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))


def _recorder_with_fake_cursor(monkeypatch, pricing=None):
    cur = _FakeCursor()

    @asynccontextmanager
    async def _fake_get_cursor(pool):
        yield cur

    monkeypatch.setattr(usage_mod, "get_cursor", _fake_get_cursor)
    return UsageRecorder(pool=object(), source="api", pricing=pricing or {}), cur


async def test_record_writes_row_with_cost(monkeypatch):
    rec_obj, cur = _recorder_with_fake_cursor(
        monkeypatch, pricing={"m": {"input": 2.0, "output": 8.0}}
    )
    rec_obj.record(CallRecord(
        call_type="chat", model="m", status="success",
        attempts=1, latency_ms=42, input_tokens=1000, output_tokens=500,
    ))
    await asyncio.sleep(0)  # 让 fire-and-forget task 跑完
    await asyncio.sleep(0)
    assert len(cur.executed) == 1
    _, params = cur.executed[0]
    assert params["source"] == "api"
    assert params["cost"] == 0.006
    assert params["status"] == "success"


async def test_write_failure_never_propagates(monkeypatch):
    @asynccontextmanager
    async def _broken_get_cursor(pool):
        raise ConnectionError("pg down")
        yield  # pragma: no cover

    monkeypatch.setattr(usage_mod, "get_cursor", _broken_get_cursor)
    recorder = UsageRecorder(pool=object(), source="api", pricing={})
    recorder.record(CallRecord(
        call_type="chat", model="m", status="failed", attempts=3, latency_ms=1,
    ))
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # 不抛即通过(fail-open)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_governance_usage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.governance.usage'`

- [ ] **Step 3: Write implementation**

在 `rag/common/logging.py` 的 `reset_session` 函数之后添加:

```python
def current_session_id() -> str | None:
    """读取当前异步上下文绑定的 session_id(无则 None)。供治理层统计记录使用。"""
    return _session_id_var.get()
```

创建 `rag/governance/usage.py`:

```python
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
```

创建 `alembic/versions/0010_llm_call_log.py`:

```python
"""llm_call_log - LLM 调用逐次统计(治理层)

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-17
"""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_call_log (
            id            BIGSERIAL PRIMARY KEY,
            created_at    timestamptz NOT NULL DEFAULT now(),
            call_type     text NOT NULL,
            model         text NOT NULL,
            source        text NOT NULL,
            session_id    text,
            status        text NOT NULL,
            error_type    text,
            attempts      int NOT NULL,
            latency_ms    int NOT NULL,
            input_tokens  int,
            output_tokens int,
            cost          numeric(12, 6)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_call_log_created_at ON llm_call_log (created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_call_log_model_created"
        " ON llm_call_log (model, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS llm_call_log")
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_governance_usage.py tests/test_logging.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Apply migration(需 docker compose postgres 在跑;没起则跳过,标注在 PR 描述)**

Run: `docker compose up -d postgres`(已在跑则跳过)然后 `uv run alembic upgrade head`
Expected: `Running upgrade 0009 -> 0010, llm_call_log - LLM 调用逐次统计(治理层)`

- [ ] **Step 6: Commit**

```bash
git add rag/common/logging.py rag/governance/usage.py alembic/versions/0010_llm_call_log.py tests/test_governance_usage.py
git commit -m "feat(governance): llm_call_log 迁移与 UsageRecorder 异步统计写入"
```

---

### Task 6: LLMGuard 组合(rag/governance/guard.py + __init__.py)

**Files:**
- Create: `rag/governance/guard.py`
- Modify: `rag/governance/__init__.py`
- Test: `tests/test_governance_guard.py`(新建)

**Interfaces:**
- Consumes: Task 1-5 全部产物
- Produces:
  - `CallTracker` dataclass — 字段:`call_type: str, model: str, attempts: int = 0, input_tokens: int | None = None, output_tokens: int | None = None`;方法 `set_tokens(input_tokens: int | None, output_tokens: int | None) -> None`
  - `LLMGuard(limiter, breaker, recorder, settings)`:
    - `acquire(quota: str)` — async context manager:熔断检查 → 限流获取;body 异常自动上报熔断器;供应商 429 触发冷却;finally 释放坑位
    - `track(call_type: str, model: str)` — async context manager,yield `CallTracker`;退出时按异常类型定 status(GovernanceRejectedError→rejected,其他异常→failed,正常→success)并 `recorder.record(...)`
  - `create_guard(redis, pool, source: str) -> LLMGuard | None`(`rag/governance/__init__.py`;`GOVERNANCE_ENABLED=false` 时返回 None)

- [ ] **Step 1: Write the failing test**

创建 `tests/test_governance_guard.py`:

```python
import pytest

from rag.common.exception import (
    CircuitOpenError,
    LLMRateLimitError,
    LLMServerError,
    RateLimitExceededError,
)
from rag.governance.config import GovernanceSettings
from rag.governance.guard import CallTracker, LLMGuard
from rag.governance.usage import CallRecord


class _FakeLimiter:
    def __init__(self, reject=False):
        self.reject = reject
        self.released = []
        self.cooldowns = []

    async def acquire(self, quota):
        if self.reject:
            raise RateLimitExceededError("限流")
        return "member-1"

    async def release(self, quota, member):
        self.released.append((quota, member))

    async def start_cooldown(self, quota, seconds=None):
        self.cooldowns.append((quota, seconds))


class _FakeBreaker:
    def __init__(self, probe=None, open_=False):
        self.probe = probe
        self.open_ = open_
        self.successes = []
        self.failures = []
        self.probe_released = []

    async def check(self, quota):
        if self.open_:
            raise CircuitOpenError("熔断")
        return self.probe

    async def record_success(self, quota, probe):
        self.successes.append((quota, probe))

    async def record_failure(self, quota, probe):
        self.failures.append((quota, probe))

    async def release_probe(self, quota):
        self.probe_released.append(quota)


class _FakeRecorder:
    def __init__(self):
        self.records: list[CallRecord] = []

    def record(self, rec):
        self.records.append(rec)


def _guard(limiter=None, breaker=None, recorder=None):
    return LLMGuard(
        limiter=limiter or _FakeLimiter(),
        breaker=breaker or _FakeBreaker(),
        recorder=recorder or _FakeRecorder(),
        settings=GovernanceSettings(_env_file=None),
    )


async def test_acquire_success_reports_breaker_and_releases():
    limiter, breaker = _FakeLimiter(), _FakeBreaker()
    async with _guard(limiter, breaker).acquire("chat"):
        pass
    assert breaker.successes == [("chat", None)]
    assert limiter.released == [("chat", "member-1")]


async def test_acquire_retryable_failure_counts_to_breaker():
    limiter, breaker = _FakeLimiter(), _FakeBreaker()
    with pytest.raises(LLMServerError):
        async with _guard(limiter, breaker).acquire("chat"):
            raise LLMServerError("500", status_code=500)
    assert breaker.failures == [("chat", None)]
    assert limiter.released == [("chat", "member-1")]


async def test_provider_429_starts_cooldown_not_breaker():
    limiter, breaker = _FakeLimiter(), _FakeBreaker()
    with pytest.raises(LLMRateLimitError):
        async with _guard(limiter, breaker).acquire("chat"):
            raise LLMRateLimitError("429", status_code=429)
    assert len(limiter.cooldowns) == 1
    assert breaker.failures == []  # 429 不计入熔断


async def test_limiter_rejection_releases_probe():
    limiter, breaker = _FakeLimiter(reject=True), _FakeBreaker(probe="probe")
    with pytest.raises(RateLimitExceededError):
        async with _guard(limiter, breaker).acquire("chat"):
            pass  # pragma: no cover
    assert breaker.probe_released == ["chat"]


async def test_track_records_success_with_tokens():
    recorder = _FakeRecorder()
    async with _guard(recorder=recorder).track("chat", "m") as t:
        t.attempts += 1
        t.set_tokens(100, 50)
    rec = recorder.records[0]
    assert (rec.status, rec.attempts, rec.input_tokens, rec.output_tokens) == (
        "success", 1, 100, 50,
    )
    assert rec.latency_ms >= 0


async def test_track_records_rejected_status():
    recorder = _FakeRecorder()
    with pytest.raises(CircuitOpenError):
        async with _guard(recorder=recorder).track("chat", "m") as t:
            t.attempts += 1
            raise CircuitOpenError("熔断")
    rec = recorder.records[0]
    assert rec.status == "rejected"
    assert rec.error_type == "CircuitOpenError"


async def test_create_guard_disabled_returns_none(monkeypatch):
    from rag.governance import create_guard
    from rag.governance.config import get_governance_settings

    # 环境变量优先级高于 .env 文件:保证本机 .env 开了 GOVERNANCE_ENABLED 也不影响本测试
    monkeypatch.setenv("GOVERNANCE_ENABLED", "false")
    get_governance_settings.cache_clear()
    try:
        assert create_guard(object(), object(), source="api") is None
    finally:
        get_governance_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_governance_guard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.governance.guard'`

- [ ] **Step 3: Write implementation**

创建 `rag/governance/guard.py`:

```python
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from rag.common.exception import (
    GovernanceRejectedError,
    LLMRateLimitError,
    is_retryable,
)
from rag.common.logging import current_session_id, get_logger
from rag.governance.breaker import RedisCircuitBreaker
from rag.governance.config import GovernanceSettings
from rag.governance.limiter import RedisRateLimiter
from rag.governance.usage import CallRecord, UsageRecorder

logger = get_logger()


def _is_provider_429(exc: BaseException) -> bool:
    """供应商侧 429(触发全局冷却)。本地限流拒绝不算(那是自己拒的)。"""
    if isinstance(exc, GovernanceRejectedError):
        return False
    if isinstance(exc, LLMRateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


def _retry_after_seconds(exc: BaseException) -> float | None:
    """从异常链提取 Retry-After 头(秒);拿不到返回 None(用默认冷却)。"""
    for e in (exc, exc.__cause__):
        rsp = getattr(e, "response", None)
        headers = getattr(rsp, "headers", None)
        if headers is not None:
            ra = headers.get("retry-after")
            if ra is not None and str(ra).isdigit():
                return float(ra)
    return None


@dataclass
class CallTracker:
    """一次逻辑调用的可变跟踪器:重试循环内累加 attempts,拿到响应后填 tokens。"""

    call_type: str
    model: str
    attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def set_tokens(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMGuard:
    """LLM 调用治理:限流 + 熔断(acquire,每次尝试一次)与统计(track,每次逻辑调用一次)。"""

    def __init__(
        self,
        limiter: RedisRateLimiter,
        breaker: RedisCircuitBreaker,
        recorder: UsageRecorder,
        settings: GovernanceSettings,
    ) -> None:
        self._limiter = limiter
        self._breaker = breaker
        self._recorder = recorder
        self._settings = settings

    @asynccontextmanager
    async def acquire(self, quota: str):
        """包住单次供应商调用:熔断检查 → 限流获取 → 执行 → 结果上报。"""
        probe = await self._breaker.check(quota)  # CircuitOpenError 直接外抛
        try:
            member = await self._limiter.acquire(quota)  # RateLimitExceededError 外抛
        except BaseException:
            if probe:
                # 探针请求没能发出去,归还探针让别人探
                await self._breaker.release_probe(quota)
            raise
        try:
            yield
        except BaseException as e:
            if is_retryable(e):
                # 可重试集合(5xx/网络/超时)与熔断计数集合一致
                await self._breaker.record_failure(quota, probe)
            if _is_provider_429(e):
                await self._limiter.start_cooldown(quota, _retry_after_seconds(e))
            raise
        else:
            await self._breaker.record_success(quota, probe)
        finally:
            await self._limiter.release(quota, member)

    @asynccontextmanager
    async def track(self, call_type: str, model: str):
        """包住整个逻辑调用(重试循环之外):计时并生成一行 CallRecord。"""
        tracker = CallTracker(call_type=call_type, model=model)
        status = "success"
        error_type: str | None = None
        t0 = time.perf_counter()
        try:
            yield tracker
        except BaseException as e:
            status = "rejected" if isinstance(e, GovernanceRejectedError) else "failed"
            error_type = type(e).__name__
            raise
        finally:
            self._recorder.record(CallRecord(
                call_type=call_type,
                model=model,
                status=status,
                attempts=tracker.attempts,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                session_id=current_session_id(),
                error_type=error_type,
                input_tokens=tracker.input_tokens,
                output_tokens=tracker.output_tokens,
            ))
```

替换 `rag/governance/__init__.py` 内容:

```python
from rag.governance.breaker import RedisCircuitBreaker
from rag.governance.config import GovernanceSettings, get_governance_settings
from rag.governance.guard import LLMGuard
from rag.governance.limiter import RedisRateLimiter
from rag.governance.usage import UsageRecorder

__all__ = [
    "GovernanceSettings",
    "LLMGuard",
    "create_guard",
    "get_governance_settings",
]


def create_guard(redis, pool, source: str) -> LLMGuard | None:
    """按配置组装 LLMGuard。GOVERNANCE_ENABLED=false 返回 None(模型层零开销直通)。

    source: api | worker | eval,写入 llm_call_log.source 列。
    """
    settings = get_governance_settings()
    if not settings.GOVERNANCE_ENABLED:
        return None
    return LLMGuard(
        limiter=RedisRateLimiter(redis, settings),
        breaker=RedisCircuitBreaker(redis, settings),
        recorder=UsageRecorder(pool, source, settings.pricing()),
        settings=settings,
    )
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_governance_guard.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/governance/guard.py rag/governance/__init__.py tests/test_governance_guard.py
git commit -m "feat(governance): LLMGuard 组合限流/熔断/统计与 create_guard 工厂"
```

---

### Task 7: NormalModel 织入 guard 与超时(rag/models/normal.py)

**Files:**
- Modify: `rag/models/normal.py`
- Test: `tests/test_normal.py`(追加)

**Interfaces:**
- Consumes: `LLMGuard.acquire/track`、`CallTracker`(Task 6)、`get_governance_settings`(Task 2)、`LLMTimeoutError`(Task 1)
- Produces: `NormalModel(settings, guard: LLMGuard | None = None)` — 对外方法签名不变(`ainvoke` / `ainvoke_structured` / `astream`);guard=None 时行为与现状完全一致

**织入规则(所有方法一致):** `track`(逻辑调用级)在最外;`AsyncRetrying` 在中;`acquire`(尝试级)在每次尝试内;`_translate` 最内。attempts 在每次尝试开始时 +1。

- [ ] **Step 1: Write the failing test**

在 `tests/test_normal.py` 末尾追加:

```python
# ── 治理层织入 ──────────────────────────────────────────

from contextlib import asynccontextmanager

from rag.common.exception import LLMServerError
from rag.governance.guard import CallTracker


class _FakeGuard:
    """记录 acquire/track 调用的假 guard。"""

    def __init__(self):
        self.acquired: list[str] = []
        self.trackers: list[CallTracker] = []

    @asynccontextmanager
    async def acquire(self, quota):
        self.acquired.append(quota)
        yield

    @asynccontextmanager
    async def track(self, call_type, model):
        t = CallTracker(call_type=call_type, model=model)
        self.trackers.append(t)
        yield t


class _FakeMsgWithUsage:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 100, "output_tokens": 50}


class _FakeModelWithUsage:
    async def ainvoke(self, messages):
        return _FakeMsgWithUsage("answer")


async def test_guard_weave_records_quota_and_tokens():
    guard = _FakeGuard()
    m = NormalModel(Settings(), guard=guard)
    m._model = _FakeModelWithUsage()
    assert await m.ainvoke(["hi"]) == "answer"
    assert guard.acquired == ["chat"]
    t = guard.trackers[0]
    assert (t.call_type, t.attempts) == ("chat", 1)
    assert (t.input_tokens, t.output_tokens) == (100, 50)


async def test_guard_weave_retry_acquires_per_attempt(monkeypatch):
    monkeypatch.setattr(normal, "wait_exponential_jitter", lambda **kw: wait_none())
    guard = _FakeGuard()

    class _FailOnceModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise LLMServerError("500", status_code=500)
            return _FakeMsgWithUsage("ok")

    m = NormalModel(Settings(), guard=guard)
    m._model = _FailOnceModel()
    assert await m.ainvoke(["hi"]) == "ok"
    assert guard.acquired == ["chat", "chat"]  # 每次尝试各过一次 guard
    assert guard.trackers[0].attempts == 2


async def test_guard_none_keeps_passthrough():
    m = NormalModel(Settings())
    m._model = _FakeModel()
    assert await m.ainvoke(["hi"]) == "answer"  # 无 guard 完全直通


async def test_astream_weave_uses_chat_stream_call_type():
    guard = _FakeGuard()

    class _StreamOnly:
        def astream(self, messages):
            async def _gen():
                yield _FakeMsgWithUsage("tok")
            return _gen()

    m = NormalModel(Settings(), guard=guard)
    m._model = _StreamOnly()
    tokens = [c.content async for c in m.astream(["hi"])]
    assert tokens == ["tok"]
    t = guard.trackers[0]
    assert t.call_type == "chat_stream"
    assert (t.input_tokens, t.output_tokens) == (100, 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_normal.py -v`
Expected: 新增 4 个测试 FAIL — `TypeError: NormalModel.__init__() got an unexpected keyword argument 'guard'`;既有测试 PASS

- [ ] **Step 3: Write implementation**

修改 `rag/models/normal.py`。头部 import 区:新增 `import asyncio`(放在 `import json` 之前),并把既有的 `from contextlib import asynccontextmanager` 改为:

```python
from contextlib import asynccontextmanager, nullcontext
```

新增异常导入与治理导入(替换原 `from rag.common.exception import ...` 行):

```python
from rag.common.exception import (
    LLMException,
    LLMTimeoutError,
    from_http_error,
    is_retryable,
)
from rag.governance.config import get_governance_settings
from rag.governance.guard import LLMGuard
```

`__init__` 替换为:

```python
    def __init__(self, settings: Settings, guard: LLMGuard | None = None):
        gov = get_governance_settings()
        self._model = ChatOpenAI(
            api_key=settings.MODEL_KEY,
            model=settings.MODEL_NAME,
            base_url=settings.MODEL_URL,
            temperature=0,
            seed=42,
            timeout=gov.LLM_TIMEOUT_SECONDS,  # 显式超时:流式为逐 chunk 读超时
            stream_usage=True,  # 流式末 chunk 携带 usage,供成本统计
        )
        self._model_name = settings.MODEL_NAME
        self._guard = guard
        self._timeout = gov.LLM_TIMEOUT_SECONDS
```

在 `bind_tools` 之后新增两个私有辅助方法与模块级 usage 提取函数(函数放在 `_extract_status_code` 之后):

```python
def _usage_from(msg) -> tuple[int | None, int | None]:
    """从 LangChain 消息取 token 用量;无 usage_metadata 时返回 (None, None)。"""
    usage = getattr(msg, "usage_metadata", None) or {}
    return usage.get("input_tokens"), usage.get("output_tokens")
```

```python
    def _acquire(self, quota: str):
        """guard 未注入时零开销直通(null-object 模式)。"""
        if self._guard is None:
            return nullcontext()
        return self._guard.acquire(quota)

    def _track(self, call_type: str):
        """逻辑调用级统计;guard 未注入时 yield None。"""
        if self._guard is None:
            return nullcontext()
        return self._guard.track(call_type, self._model_name)
```

`_translate` 增加超时翻译分支(整体替换):

```python
    @asynccontextmanager
    async def _translate(self):
        try:
            yield
        except LLMException:
            raise
        except TimeoutError as e:
            # asyncio.timeout 兜底触发;可重试、计入熔断
            raise LLMTimeoutError("LLM 调用超时", model=self._model_name) from e
        except Exception as e:
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise
```

`ainvoke` 整体替换:

```python
    async def ainvoke(self, messages: list[BaseMessage | str]) -> str:
        """带重试的异步调用。track 逻辑调用级,acquire 每次尝试级。"""
        async with self._track("chat") as tracker:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
                retry=retry_if_exception(is_retryable),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    if tracker is not None:
                        tracker.attempts += 1
                    async with self._acquire("chat"):
                        async with self._translate():
                            async with asyncio.timeout(self._timeout):
                                rsp = await self._model.ainvoke(messages)
                            if tracker is not None:
                                tracker.set_tokens(*_usage_from(rsp))
                            return rsp.content
            raise AssertionError("unreachable")
```

`ainvoke_structured` 的重试循环段替换(schema 注入部分不动;json_mode 丢失 usage 元数据,tokens 记 NULL 属预期):

```python
        async with self._track("chat_structured") as tracker:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
                retry=retry_if_exception(_is_retryable_structured),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    if tracker is not None:
                        tracker.attempts += 1
                    async with self._acquire("chat"):
                        async with self._translate():
                            async with asyncio.timeout(self._timeout):
                                result = await structured.ainvoke(msgs)
                            if result is None:
                                raise OutputParserException("模型未返回结构化输出")
                            return result
            raise AssertionError("unreachable")
```

`astream` 整体替换(不重试维持现状;SDK 逐 chunk 读超时兜底,不加 asyncio.timeout 以免误杀长回答):

```python
    async def astream(self, messages: list[BaseMessage | str]):
        """单次异步流式(不重试,流式中途重试语义复杂,维持现状)。"""
        async with self._track("chat_stream") as tracker:
            if tracker is not None:
                tracker.attempts = 1
            async with self._acquire("chat"):
                async with self._translate():
                    async for chunk in self._model.astream(messages):
                        usage = getattr(chunk, "usage_metadata", None)
                        if tracker is not None and usage:
                            tracker.set_tokens(
                                usage.get("input_tokens"), usage.get("output_tokens")
                            )
                        yield chunk
```

- [ ] **Step 4: Run test to verify pass(全量回归)**

Run: `uv run pytest tests/test_normal.py tests/test_nodes.py tests/test_workflow.py -v`
Expected: 全部 PASS(guard=None 直通保证既有测试不动)

- [ ] **Step 5: Commit**

```bash
git add rag/models/normal.py tests/test_normal.py
git commit -m "feat(governance): NormalModel 织入 guard - 限流/熔断/超时/统计"
```

---

### Task 8: EmbeddingModel 与 QwenReranker 织入(rag/models/embedding.py, rerank.py)

**Files:**
- Modify: `rag/models/embedding.py`
- Modify: `rag/models/rerank.py`
- Test: `tests/test_embedding.py`(追加)、`tests/test_rerank_guard.py`(新建)

**Interfaces:**
- Consumes: Task 6 的 `LLMGuard`、Task 2 的 `get_governance_settings`、Task 1 的 `LLMTimeoutError`、已有 `from_http_error`
- Produces:
  - `EmbeddingModel(settings, guard: LLMGuard | None = None)` — `embed` 签名不变
  - `QwenReranker(settings, guard: LLMGuard | None = None)` — `rerank` 签名不变;治理拒绝/熔断也走既有降级(返回原序 chunks)

- [ ] **Step 1: Write the failing tests**

在 `tests/test_embedding.py` 末尾追加(该文件已有 `_FakeEmbeddingsAPI` 类似的假客户端则复用其风格;若无,按下面自建):

```python
# ── 治理层织入 ──────────────────────────────────────────

from contextlib import asynccontextmanager

from rag.governance.guard import CallTracker


class _GuardSpy:
    def __init__(self):
        self.acquired: list[str] = []
        self.trackers: list[CallTracker] = []

    @asynccontextmanager
    async def acquire(self, quota):
        self.acquired.append(quota)
        yield

    @asynccontextmanager
    async def track(self, call_type, model):
        t = CallTracker(call_type=call_type, model=model)
        self.trackers.append(t)
        yield t


class _FakeUsage:
    prompt_tokens = 7


class _FakeEmbData:
    def __init__(self, index):
        self.index = index
        self.embedding = [0.0] * 4


class _FakeEmbResponse:
    data = [_FakeEmbData(0)]
    usage = _FakeUsage()


class _FakeEmbeddingsClient:
    class embeddings:  # noqa: N801 - 模仿 openai SDK 结构
        @staticmethod
        async def create(**kw):
            return _FakeEmbResponse()


async def test_embed_guard_weave_records_usage():
    from rag.config import Settings
    from rag.models.embedding import EmbeddingModel

    guard = _GuardSpy()
    m = EmbeddingModel(Settings(), guard=guard)
    m._client = _FakeEmbeddingsClient()
    result = await m.embed(["文本"])
    assert len(result) == 1
    assert guard.acquired == ["embedding"]
    t = guard.trackers[0]
    assert (t.call_type, t.input_tokens) == ("embedding", 7)
```

创建 `tests/test_rerank_guard.py`:

```python
from contextlib import asynccontextmanager

import httpx
import pytest

from rag.common.exception import CircuitOpenError
from rag.config import Settings
from rag.governance.guard import CallTracker
from rag.models.rerank import QwenReranker


class _GuardSpy:
    def __init__(self, open_=False):
        self.open_ = open_
        self.acquired: list[str] = []
        self.trackers: list[CallTracker] = []

    @asynccontextmanager
    async def acquire(self, quota):
        self.acquired.append(quota)
        if self.open_:
            raise CircuitOpenError("熔断")
        yield

    @asynccontextmanager
    async def track(self, call_type, model):
        t = CallTracker(call_type=call_type, model=model)
        self.trackers.append(t)
        yield t


def _reranker(guard=None) -> QwenReranker:
    settings = Settings(RERANK_BASE_URL="http://mock-rerank.local", RERANK_KEY="k")
    return QwenReranker(settings, guard=guard)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


async def test_rerank_guard_weave(monkeypatch):
    guard = _GuardSpy()
    r = _reranker(guard=guard)

    async def _fake_post(url, json=None):
        return _FakeResponse({"results": [{"index": 0, "relevance_score": 0.9}]})

    monkeypatch.setattr(r._client, "post", _fake_post)
    chunks = [{"text": "a"}]
    result = await r.rerank("q", chunks)
    assert result[0]["rerank_score"] == 0.9
    assert guard.acquired == ["rerank"]
    assert guard.trackers[0].call_type == "rerank"


async def test_rerank_circuit_open_degrades_to_original_order():
    guard = _GuardSpy(open_=True)
    r = _reranker(guard=guard)
    chunks = [{"text": "a"}, {"text": "b"}]
    assert await r.rerank("q", chunks) == chunks  # 熔断拒绝也降级,不打断 chat


async def test_rerank_5xx_translated_inside_guard(monkeypatch):
    """5xx 在 guard 作用域内被翻译成 LLM 异常(可计入熔断),对外仍降级。"""
    guard = _GuardSpy()
    r = _reranker(guard=guard)

    class _ErrResponse:
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500", request=httpx.Request("POST", "http://x"),
                response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
            )

    async def _fake_post(url, json=None):
        return _ErrResponse()

    monkeypatch.setattr(r._client, "post", _fake_post)
    chunks = [{"text": "a"}]
    assert await r.rerank("q", chunks) == chunks  # 降级
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding.py tests/test_rerank_guard.py -v`
Expected: 新测试 FAIL — `TypeError: ... unexpected keyword argument 'guard'`

- [ ] **Step 3: Write implementation**

修改 `rag/models/embedding.py`(整体替换为):

```python
import logging
from contextlib import nullcontext

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.common.exception import is_retryable
from rag.common.logging import get_logger
from rag.config import Settings
from rag.governance.config import get_governance_settings
from rag.governance.guard import LLMGuard

logger = get_logger()


class EmbeddingModel:
    """异步 embedding 模型,带重试与超时;guard 注入后纳入限流/熔断/统计。"""

    def __init__(self, settings: Settings, guard: LLMGuard | None = None):
        gov = get_governance_settings()
        self._client = AsyncOpenAI(
            api_key=settings.EMBEDDING_KEY,
            base_url=settings.EMBEDDING_URL,
            timeout=gov.EMBEDDING_TIMEOUT_SECONDS,
        )
        self._model = settings.EMBEDDING_MODEL
        self._dim = settings.EMBEDDING_DIM
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._guard = guard

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _acquire(self):
        if self._guard is None:
            return nullcontext()
        return self._guard.acquire("embedding")

    def _track(self):
        if self._guard is None:
            return nullcontext()
        return self._guard.track("embedding", self._model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self._track() as tracker:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
                retry=retry_if_exception(is_retryable),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    if tracker is not None:
                        tracker.attempts += 1
                    async with self._acquire():
                        rsp = await self._client.embeddings.create(
                            model=self._model,
                            input=texts,
                            dimensions=self._dim,
                            encoding_format="float",
                        )
            usage = getattr(rsp, "usage", None)
            if tracker is not None and usage is not None:
                # embedding 只有输入侧 token
                tracker.set_tokens(getattr(usage, "prompt_tokens", None), None)
        ordered = sorted(rsp.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]
```

修改 `rag/models/rerank.py`:头部 import 区替换为:

```python
from contextlib import nullcontext

from pydantic import BaseModel, Field

from rag.common.exception import LLMTimeoutError, from_http_error
from rag.common.logging import get_logger
from rag.config import Settings
from rag.governance.config import get_governance_settings
from rag.governance.guard import LLMGuard
from rag.models.base import ChatModel
```

`QwenReranker.__init__` 替换为:

```python
    def __init__(self, settings: Settings, guard: LLMGuard | None = None):
        import httpx

        if not settings.RERANK_BASE_URL:
            raise ValueError("RERANK_BASE_URL 未配置，无法初始化 QwenReranker")
        gov = get_governance_settings()
        self._client = httpx.AsyncClient(
            base_url=settings.RERANK_BASE_URL.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.RERANK_KEY or settings.EMBEDDING_KEY}",
                "Content-Type": "application/json",
            },
            timeout=gov.RERANK_TIMEOUT_SECONDS,
        )
        self._model = settings.RERANK_MODEL
        self._guard = guard
```

在 `__init__` 后新增私有方法:

```python
    def _acquire(self):
        if self._guard is None:
            return nullcontext()
        return self._guard.acquire("rerank")

    def _track(self):
        if self._guard is None:
            return nullcontext()
        return self._guard.track("rerank", self._model)

    async def _call_api(self, body: dict) -> dict:
        """真正的 HTTP 调用:guard 作用域内把 httpx 异常翻译为 LLM 异常体系,
        使 5xx/超时计入熔断、429 触发冷却。"""
        import httpx

        async with self._track() as tracker:
            if tracker is not None:
                tracker.attempts = 1
            async with self._acquire():
                try:
                    rsp = await self._client.post("/v1/reranks", json=body)
                    rsp.raise_for_status()
                    return rsp.json()
                except httpx.TimeoutException as e:
                    raise LLMTimeoutError("rerank 调用超时", model=self._model) from e
                except httpx.HTTPStatusError as e:
                    raise from_http_error(
                        e.response.status_code, str(e), model=self._model
                    ) from e
```

`rerank` 方法中原 `try: rsp = await self._client.post(...) ... except Exception:` 段替换为:

```python
        try:
            data = await self._call_api(body)
        except Exception:
            logger.warning("QwenRerank API 调用失败，降级为原始召回顺序", exc_info=True)
            return chunks
```

(方法其余部分 —— `results` 解析、score_map 排序 —— 不动。)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_embedding.py tests/test_rerank_guard.py tests/test_retriever.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/models/embedding.py rag/models/rerank.py tests/test_embedding.py tests/test_rerank_guard.py
git commit -m "feat(governance): EmbeddingModel 与 QwenReranker 织入 guard"
```

---

### Task 9: API/Worker 注入 + chat 友好报错 + .env.example

**Files:**
- Modify: `rag/api/main.py`(lifespan 内、redis 初始化之后)
- Modify: `rag/worker/main.py`(on_startup / on_shutdown / WorkerCtx)
- Modify: `rag/api/modules/chat/service.py:69`(`stream.error` 行)
- Modify: `.env.example`(追加治理配置块)
- Test: `tests/test_chat_service_errors.py`(新建)

**Interfaces:**
- Consumes: `create_guard`(Task 6)、`friendly_message`(Task 1)、`create_redis_client`(`rag.db.redis`,已有)
- Produces: `app.state.guard`、Worker `ctx["redis"]` / `ctx["guard"]`;chat SSE 错误事件只含友好文案

- [ ] **Step 1: Write the failing test**

创建 `tests/test_chat_service_errors.py`:

```python
"""chat producer 的异常必须映射为友好文案,原始异常不外泄到 SSE。"""
import json

from rag.api.modules.chat import service
from rag.common.exception import CircuitOpenError


class _ExplodingLLM:
    """任何调用都抛熔断异常的假 LLM(满足 ChatModel 用到的接口)。"""

    async def ainvoke(self, messages):
        raise CircuitOpenError("gov:cb:chat open")

    async def ainvoke_structured(self, messages, schema):
        raise CircuitOpenError("gov:cb:chat open")

    def astream(self, messages):
        async def _gen():
            raise CircuitOpenError("gov:cb:chat open")
            yield  # pragma: no cover
        return _gen()


class _NoopMemory:
    async def search(self, session_id, query, top_k=5, filters=None):
        return []

    async def get_recent_messages(self, session_id, n=10):
        return []

    async def add_message(self, session_id, text, metadata=None):
        pass

    async def persist_turn(self, session_id, query, answer):
        pass


class _NoopRetriever:
    async def search(self, query, knowledge_base_ids=None, top_k=5):
        return []

    async def fetch_parent_contents(self, document_ids):
        return {}


async def test_stream_error_event_is_friendly_not_raw():
    lines = []
    async for sse in service.stream_chat(
        "s1", "你好",
        llm=_ExplodingLLM(), memory_manager=_NoopMemory(),
        retriever=_NoopRetriever(), reranker=None, pool=None,
    ):
        lines.append(sse)
    error_events = [
        json.loads(l.removeprefix("data: "))
        for l in lines
        if l.startswith("data: {") and '"error"' in l
    ]
    assert error_events, "应产生 error 事件"
    assert error_events[0]["data"] == "AI 服务暂时不可用,请稍后重试"
    assert "gov:cb" not in error_events[0]["data"]  # 原始异常信息不外泄
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_service_errors.py -v`
Expected: FAIL — error 事件 data 为原始异常字符串 `gov:cb:chat open`,断言不等

- [ ] **Step 3: Write implementation**

**(a)** `rag/api/modules/chat/service.py`:import 区加一行,`_produce` 内改一行:

```python
from rag.common.exception import friendly_message
```

`_produce` 中 `await stream.error(str(e))` 改为:

```python
                await stream.error(friendly_message(e))
```

**(b)** `rag/api/main.py`:import 区加 `from rag.governance import create_guard`。lifespan 中,`redis 连接池初始化完成` 日志之后、agent 依赖单例之前插入:

```python
        # ------ 初始化 LLM 治理层 -------
        logger.info("初始化 LLM 治理层")
        app.state.guard = create_guard(app.state.redis, pool, source="api")
        logger.info(
            "LLM 治理层%s", "已启用" if app.state.guard else "未启用(GOVERNANCE_ENABLED=false)"
        )
```

三处模型构造改为传 guard:

```python
        embedding = EmbeddingModel(settings, guard=app.state.guard)
        ...
        app.state.llm = NormalModel(settings, guard=app.state.guard)
        ...
            app.state.reranker = QwenReranker(settings, guard=app.state.guard)
```

**(c)** `rag/worker/main.py`:import 区加:

```python
from rag.db import create_redis_client
from rag.governance import create_guard
```

`WorkerCtx` 加两个字段(在 `llm: ChatModel` 之后):

```python
    redis: object
    guard: object | None
```

`on_startup` 中 `ctx["pg"] = ...` 之后插入,并改两处模型构造:

```python
    ctx["redis"] = create_redis_client(settings)
    ctx["guard"] = create_guard(ctx["redis"], ctx["pg"], source="worker")
```

```python
    ctx["embedding"] = EmbeddingModel(settings, guard=ctx["guard"])
    ...
    ctx["llm"] = NormalModel(settings, guard=ctx["guard"])
```

`on_shutdown` 加:

```python
    if ctx.get("redis") is not None:
        await ctx["redis"].aclose()
```

**(d)** `.env.example` 末尾追加:

```bash
# ── LLM 调用治理(限流/熔断/超时/成本统计) ──
GOVERNANCE_ENABLED=true
CHAT_RPM_LIMIT=60
CHAT_MAX_CONCURRENCY=8
EMBEDDING_RPM_LIMIT=500
EMBEDDING_MAX_CONCURRENCY=10
RERANK_RPM_LIMIT=120
RERANK_MAX_CONCURRENCY=8
ACQUIRE_MAX_WAIT_SECONDS=10
BREAKER_FAILURE_THRESHOLD=5
BREAKER_COOLDOWN_SECONDS=30
LLM_TIMEOUT_SECONDS=60
EMBEDDING_TIMEOUT_SECONDS=30
RERANK_TIMEOUT_SECONDS=30
# 每百万 token 单价(元),按实际模型改
LLM_PRICING={"deepseek-chat": {"input": 2.0, "output": 8.0}, "text-embedding-v4": {"input": 0.5, "output": 0}}
```

- [ ] **Step 4: Run tests to verify pass(全量单测回归)**

Run: `uv run pytest tests/ -v --ignore=tests/test_db.py --ignore=tests/test_db_neo4j.py`
Expected: 全部 PASS

- [ ] **Step 5: 冒烟启动验证**

Run: `uv run python -c "from rag.api.main import start_app; start_app(); print('app ok')"`
Expected: 输出 `app ok`(lifespan 不执行,只验证 import 链无循环依赖)

- [ ] **Step 6: Commit**

```bash
git add rag/api/main.py rag/worker/main.py rag/api/modules/chat/service.py .env.example tests/test_chat_service_errors.py
git commit -m "feat(governance): API/Worker 注入 guard,chat SSE 友好报错"
```

---

### Task 10: 成本统计聚合 API(rag/api/modules/stats/)

**Files:**
- Create: `rag/api/modules/stats/__init__.py`
- Create: `rag/api/modules/stats/controller.py`
- Create: `rag/api/modules/stats/service.py`
- Modify: `rag/api/modules/register.py`
- Test: `tests/test_stats_service.py`(新建)

**Interfaces:**
- Consumes: `get_cursor`(`rag.db.postgres`)、`llm_call_log` 表(Task 5)
- Produces:
  - `GET /stats/llm/summary?start&end&group_by=day|model|source` → `{"group_by": str, "rows": [{"bucket", "calls", "success", "failed", "rejected", "input_tokens", "output_tokens", "cost", "avg_latency_ms", "p95_latency_ms"}]}`
  - `GET /stats/llm/recent?limit=50` → `{"rows": [...llm_call_log 原始列...]}`
  - `service.llm_summary(pool, start: datetime, end: datetime, group_by: str) -> dict`
  - `service.llm_recent(pool, limit: int) -> dict`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_stats_service.py`:

```python
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

import rag.api.modules.stats.service as stats_service


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))

    async def fetchall(self):
        return self._rows


def _patch_cursor(monkeypatch, rows):
    cur = _FakeCursor(rows)

    @asynccontextmanager
    async def _fake_get_cursor(pool):
        yield cur

    monkeypatch.setattr(stats_service, "get_cursor", _fake_get_cursor)
    return cur


async def test_summary_group_by_day_builds_bucket_sql(monkeypatch):
    cur = _patch_cursor(monkeypatch, [{"bucket": "2026-07-17", "calls": 3}])
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 18, tzinfo=timezone.utc)
    result = await stats_service.llm_summary(object(), start, end, "day")
    sql, params = cur.executed[0]
    assert "date_trunc('day', created_at)" in sql
    assert params == {"start": start, "end": end}
    assert result == {"group_by": "day", "rows": [{"bucket": "2026-07-17", "calls": 3}]}


async def test_summary_group_by_model(monkeypatch):
    cur = _patch_cursor(monkeypatch, [])
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 18, tzinfo=timezone.utc)
    await stats_service.llm_summary(object(), start, end, "model")
    sql, _ = cur.executed[0]
    assert "GROUP BY bucket" in sql
    assert "model AS bucket" in sql


async def test_summary_rejects_unknown_group_by(monkeypatch):
    _patch_cursor(monkeypatch, [])
    with pytest.raises(ValueError):
        await stats_service.llm_summary(
            object(),
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
            "user; DROP TABLE llm_call_log",  # 注入尝试必须被白名单挡下
        )


async def test_recent_clamps_limit(monkeypatch):
    cur = _patch_cursor(monkeypatch, [])
    await stats_service.llm_recent(object(), limit=9999)
    _, params = cur.executed[0]
    assert params == {"limit": 200}  # 上限 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stats_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.api.modules.stats'`

- [ ] **Step 3: Write implementation**

创建 `rag/api/modules/stats/__init__.py`:

```python
from rag.api.modules.stats.controller import stats_router

__all__ = ["stats_router"]
```

创建 `rag/api/modules/stats/service.py`:

```python
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
```

创建 `rag/api/modules/stats/controller.py`:

```python
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
```

修改 `rag/api/modules/register.py`(整体替换):

```python
from fastapi import FastAPI

from rag.api.modules.chat import chat_router
from rag.api.modules.document import document_router
from rag.api.modules.files import files_router
from rag.api.modules.health import health_router
from rag.api.modules.stats import stats_router


def register_modules(app: FastAPI):
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(document_router)
    app.include_router(files_router)
    app.include_router(stats_router)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_stats_service.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/api/modules/stats/ rag/api/modules/register.py tests/test_stats_service.py
git commit -m "feat(stats): llm_call_log 聚合查询 API - summary/recent"
```

---

### Task 11: 前端统计页 StatsPage

**Files:**
- Modify: `frontend/src/types.ts`(追加统计类型)
- Modify: `frontend/src/api/client.ts`(追加 Stats API)
- Create: `frontend/src/pages/StatsPage.tsx`
- Modify: `frontend/src/App.tsx`(加路由)
- Modify: `frontend/src/components/Layout.tsx`(NAV 加入口)

**Interfaces:**
- Consumes: Task 10 的 `/api/stats/llm/summary`、`/api/stats/llm/recent`(前端经 vite 代理 `/api` 前缀访问)
- Produces: `/stats` 路由页面

**注意:** 实现本任务写图表代码前,按 dataviz 技能的触发规则先读该技能(本任务的"趋势条"属于图表)。下面代码是布局与数据流基线,视觉规范以 dataviz 技能为准微调。

- [ ] **Step 1: 追加类型**

`frontend/src/types.ts` 末尾追加:

```typescript
// ── LLM 调用统计 ────────────────────────────────────
export interface LlmSummaryRow {
  bucket: string;
  calls: number;
  success: number;
  failed: number;
  rejected: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
  avg_latency_ms: number;
  p95_latency_ms: number | null;
}

export interface LlmSummaryResponse {
  group_by: "day" | "model" | "source";
  rows: LlmSummaryRow[];
}
```

- [ ] **Step 2: 追加 client 函数**

`frontend/src/api/client.ts` 末尾追加(import 区补 `LlmSummaryResponse`):

```typescript
// ── LLM Stats ───────────────────────────────────────

export async function getLlmSummary(
  groupBy: "day" | "model" | "source",
): Promise<LlmSummaryResponse> {
  const res = await fetch(`${BASE}/stats/llm/summary?group_by=${groupBy}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "fetch llm summary failed");
  }
  return res.json();
}
```

- [ ] **Step 3: 创建 StatsPage**

创建 `frontend/src/pages/StatsPage.tsx`:

```tsx
import { useEffect, useState } from "react";
import { getLlmSummary } from "@/api/client";
import type { LlmSummaryRow } from "@/types";

function fmtCost(v: number): string {
  return `¥${Number(v).toFixed(4)}`;
}

function fmtDay(bucket: string): string {
  // bucket 形如 "2026-07-17T00:00:00+00:00" → "07-17"
  return bucket.slice(5, 10);
}

/** 汇总卡片 */
function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xl font-semibold text-gray-100 mt-1">{value}</div>
    </div>
  );
}

/** 按天调用量横条(单色系,拒绝量叠加为警示色) */
function DayTrend({ rows }: { rows: LlmSummaryRow[] }) {
  const max = Math.max(1, ...rows.map((r) => r.calls));
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.bucket} className="flex items-center gap-2 text-xs">
          <span className="w-12 text-gray-500 shrink-0">{fmtDay(r.bucket)}</span>
          <div className="flex-1 h-4 bg-gray-800 rounded overflow-hidden flex">
            <div
              className="h-full bg-emerald-500/70"
              style={{ width: `${((r.calls - r.rejected) / max) * 100}%` }}
            />
            <div
              className="h-full bg-amber-500/70"
              style={{ width: `${(r.rejected / max) * 100}%` }}
            />
          </div>
          <span className="w-24 text-right text-gray-400 shrink-0">
            {r.calls} 次 / {fmtCost(r.cost)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function StatsPage() {
  const [byDay, setByDay] = useState<LlmSummaryRow[]>([]);
  const [byModel, setByModel] = useState<LlmSummaryRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getLlmSummary("day"), getLlmSummary("model")])
      .then(([day, model]) => {
        setByDay(day.rows);
        setByModel(model.rows);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const totals = byDay.reduce(
    (acc, r) => ({
      calls: acc.calls + r.calls,
      success: acc.success + r.success,
      rejected: acc.rejected + r.rejected,
      cost: acc.cost + Number(r.cost),
      tokens: acc.tokens + r.input_tokens + r.output_tokens,
    }),
    { calls: 0, success: 0, rejected: 0, cost: 0, tokens: 0 },
  );
  const successRate =
    totals.calls > 0 ? `${((totals.success / totals.calls) * 100).toFixed(1)}%` : "-";

  return (
    <div className="p-6 overflow-y-auto space-y-6">
      <h2 className="text-lg font-semibold">LLM 调用统计(近 14 天)</h2>
      {error && <p className="text-sm text-red-400">加载失败: {error}</p>}

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard label="总调用" value={String(totals.calls)} />
        <StatCard label="成功率" value={successRate} />
        <StatCard label="被拒绝(限流/熔断)" value={String(totals.rejected)} />
        <StatCard label="Token 合计" value={totals.tokens.toLocaleString()} />
        <StatCard label="成本合计" value={fmtCost(totals.cost)} />
      </div>

      <section className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">按天调用量与成本</h3>
        {byDay.length === 0 ? (
          <p className="text-sm text-gray-500">暂无数据</p>
        ) : (
          <DayTrend rows={byDay} />
        )}
      </section>

      <section className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">按模型分布</h3>
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-gray-500 border-b border-gray-800">
            <tr>
              <th className="py-2">模型</th>
              <th className="py-2 text-right">调用</th>
              <th className="py-2 text-right">成功/失败/拒绝</th>
              <th className="py-2 text-right">Tokens(入/出)</th>
              <th className="py-2 text-right">P95 延迟</th>
              <th className="py-2 text-right">成本</th>
            </tr>
          </thead>
          <tbody>
            {byModel.map((r) => (
              <tr key={r.bucket} className="border-b border-gray-800/50 text-gray-300">
                <td className="py-2">{r.bucket}</td>
                <td className="py-2 text-right">{r.calls}</td>
                <td className="py-2 text-right">
                  {r.success}/{r.failed}/{r.rejected}
                </td>
                <td className="py-2 text-right">
                  {r.input_tokens.toLocaleString()}/{r.output_tokens.toLocaleString()}
                </td>
                <td className="py-2 text-right">
                  {r.p95_latency_ms != null ? `${Math.round(r.p95_latency_ms)}ms` : "-"}
                </td>
                <td className="py-2 text-right">{fmtCost(r.cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
```

- [ ] **Step 4: 加路由与导航**

`frontend/src/App.tsx`:import 区加 `import StatsPage from "@/pages/StatsPage";`,`<Routes>` 内加:

```tsx
        <Route path="/stats" element={<StatsPage />} />
```

`frontend/src/components/Layout.tsx`:lucide import 加 `BarChart3`,`NAV` 数组加:

```typescript
  { to: "/stats", label: "Stats", icon: BarChart3 },
```

- [ ] **Step 5: 构建验证**

Run: `cd frontend && pnpm install && pnpm build`
Expected: `vite build` 成功,无 TypeScript 错误

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types.ts frontend/src/api/client.ts frontend/src/pages/StatsPage.tsx frontend/src/App.tsx frontend/src/components/Layout.tsx
git commit -m "feat(frontend): LLM 调用成本统计页 StatsPage"
```

---

### Task 12: 压测设施(loadtest/ mock LLM + locustfile)

**Files:**
- Modify: `pyproject.toml`(经 `uv add --dev locust`)
- Create: `loadtest/mock_llm.py`
- Create: `loadtest/locustfile.py`
- Create: `loadtest/README.md`

**Interfaces:**
- Consumes: API 的 `POST /chat/`(SSE);`.env` 中 `MODEL_URL` / `EMBEDDING_URL` 指向 mock
- Produces:
  - mock 服务器(端口 9100):`POST /v1/chat/completions`(流式+非流式)、`POST /v1/embeddings`、`POST /v1/reranks`;环境变量 `MOCK_LATENCY_MS`(默认 200)、`MOCK_ERROR_RATE`(0-1,默认 0)、`MOCK_ERROR_CODE`(默认 500)
  - locust 场景:压 `/chat/` 完整读 SSE 流

- [ ] **Step 1: Add dev dependency**

Run: `uv add --dev locust`
Expected: dev 组出现 `locust>=...`

- [ ] **Step 2: 创建 mock LLM 服务器**

创建 `loadtest/mock_llm.py`:

```python
"""OpenAI 兼容 mock 服务器 —— 压测替身,不烧真实 token。

启动: uv run uvicorn loadtest.mock_llm:app --port 9100
环境变量:
  MOCK_LATENCY_MS  响应前延迟毫秒(默认 200)
  MOCK_ERROR_RATE  错误概率 0-1(默认 0)
  MOCK_ERROR_CODE  错误时的 HTTP 状态码(默认 500)
"""
import asyncio
import json
import os
import random
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

_ANSWER_TOKENS = ["孙", "悟", "空", "三", "打", "白", "骨", "精", "。"]


def _latency_s() -> float:
    return int(os.environ.get("MOCK_LATENCY_MS", "200")) / 1000


def _should_fail() -> bool:
    return random.random() < float(os.environ.get("MOCK_ERROR_RATE", "0"))


def _error_response() -> JSONResponse:
    code = int(os.environ.get("MOCK_ERROR_CODE", "500"))
    return JSONResponse(
        status_code=code,
        content={"error": {"message": f"mock error {code}", "type": "mock"}},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    await asyncio.sleep(_latency_s())
    if _should_fail():
        return _error_response()

    created = int(time.time())
    model = body.get("model", "mock")
    if body.get("stream"):
        async def _sse():
            for tok in _ANSWER_TOKENS:
                chunk = {
                    "id": "mock", "object": "chat.completion.chunk", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)
            final = {
                "id": "mock", "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 9, "total_tokens": 109},
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")

    return {
        "id": "mock", "object": "chat.completion", "created": created, "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(_ANSWER_TOKENS)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 9, "total_tokens": 109},
    }


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    await asyncio.sleep(_latency_s())
    if _should_fail():
        return _error_response()
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    dim = int(body.get("dimensions", 1024))
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.01] * dim}
            for i in range(len(texts))
        ],
        "model": body.get("model", "mock"),
        "usage": {"prompt_tokens": 10 * len(texts), "total_tokens": 10 * len(texts)},
    }


@app.post("/v1/reranks")
async def reranks(request: Request):
    body = await request.json()
    await asyncio.sleep(_latency_s())
    if _should_fail():
        return _error_response()
    return {
        "results": [
            {"index": i, "relevance_score": 1.0 - i * 0.1}
            for i in range(len(body.get("documents", [])))
        ]
    }
```

- [ ] **Step 3: 创建 locustfile**

创建 `loadtest/locustfile.py`:

```python
"""压 /chat/ SSE 端点:发问题、完整读流、按事件类型断言。

跑法(headless 示例,S1 基线):
  uv run locust -f loadtest/locustfile.py --headless -u 10 -r 2 -t 2m --host http://localhost:8000
"""
import json
import random
import uuid

from locust import HttpUser, between, task

_QUERIES = [
    "孙悟空的金箍棒是从哪里来的?",
    "唐僧为什么要去西天取经?",
    "白骨精变了几次人形?",
    "猪八戒原来是什么神仙?",
    "火焰山是怎么形成的?",
]


class ChatUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat(self):
        payload = {"session_id": f"loadtest-{uuid.uuid4().hex[:8]}", "query": random.choice(_QUERIES)}
        got_message = False
        got_error = False
        with self.client.post(
            "/chat/", json=payload, stream=True, catch_response=True, name="/chat/ (SSE)"
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    event = json.loads(line.removeprefix("data: "))
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "message":
                    got_message = True
                elif event.get("type") == "error":
                    got_error = True
            if got_message:
                resp.success()
            elif got_error:
                # 治理层友好拒绝:对压测而言是"预期失败",单独标记
                resp.failure("degraded: SSE error event")
            else:
                resp.failure("no message events")
```

- [ ] **Step 4: 创建 README**

创建 `loadtest/README.md`:

````markdown
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
````

- [ ] **Step 5: 冒烟验证 mock**

Run: `uv run uvicorn loadtest.mock_llm:app --port 9100 &`(后台)然后
`curl -s -X POST http://localhost:9100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"m\",\"messages\":[]}"`
Expected: 返回 JSON,`choices[0].message.content` 为西游记 token 串;验证后停掉后台进程

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock loadtest/
git commit -m "feat(loadtest): mock LLM 服务器与 locust 压测场景"
```

---

### Task 13: 执行四场景压测并产出报告

**Files:**
- Create: `docs/loadtest/2026-07-17-report.md`(日期按实际执行日)
- Create: `loadtest/results/`(locust csv 输出,gitignore 可选)

**Interfaces:**
- Consumes: Task 12 的全部设施、Task 5 的 llm_call_log、Task 9 的 GOVERNANCE_ENABLED 配置

**前置:** 本任务需要人工在多终端操作(起 mock/API/locust),不适合无人值守执行;由执行者按 `loadtest/README.md` 步骤跑,每个场景结束后立即填表。

- [ ] **Step 1: 按 README 准备环境**(infra + mock + API,`.env` 指向 mock,`GOVERNANCE_ENABLED=true`)

- [ ] **Step 2: 依次执行 S1-S4**,每场景保存 locust csv(`--csv loadtest/results/s1` 等),并在场景结束后立刻跑对账 SQL 记录 status 分布

- [ ] **Step 3: 用下面模板写报告**

创建 `docs/loadtest/2026-07-17-report.md`(数字全部来自实测,禁止编造;某场景未达预期就如实记录并开后续 issue):

```markdown
# LLM 治理层负载压测报告

日期: 2026-07-17
环境: 本机(Windows 11),API 单实例,mock LLM(latency=200ms 基线),
治理配置: CHAT_RPM_LIMIT=60, CHAT_MAX_CONCURRENCY=8, BREAKER_FAILURE_THRESHOLD=5,
BREAKER_COOLDOWN_SECONDS=30, LLM_TIMEOUT_SECONDS=60

## 结果总览

| 场景 | RPS | P50 | P95 | P99 | 失败率 | 验证结论 |
|---|---|---|---|---|---|---|
| S1 基线 | | | | | | |
| S2 限流饱和 | | | | | | 限流是否生效: |
| S3 供应商故障 | | | | | | 熔断打开/恢复时间: |
| S4 慢供应商 | | | | | | 超时/坑位释放: |

## S2 限流对账(llm_call_log)

| status | count | 说明 |
|---|---|---|
| success | | |
| rejected | | 应 ≈ 超出配额的请求数 |

## S3 熔断时间线

- HH:MM:SS mock 切换 100% 500
- HH:MM:SS 日志出现「熔断器打开: chat(连续失败 5 次)」← 应在 5 次失败内
- HH:MM:SS mock 恢复
- HH:MM:SS 探针成功,熔断闭合(rejected 停止增长)

## 结论与调参建议

- (基于数据给出 CHAT_RPM_LIMIT / MAX_CONCURRENCY / 超时阈值的建议值)

## 遗留问题

- (未达预期项,逐条列出)
```

- [ ] **Step 4: 恢复 `.env`**(MODEL_URL/EMBEDDING_URL 改回真实供应商)

- [ ] **Step 5: Commit**

```bash
git add docs/loadtest/ loadtest/results/
git commit -m "docs(loadtest): LLM 治理层四场景压测报告"
```

---

## Spec 覆盖对照(自审)

| Spec 章节 | 实现任务 |
|---|---|
| 3.1 模块布局 | Task 2/3/4/5/6 |
| 3.2 注入方式 | Task 7/8/9 |
| 3.3 织入顺序 | Task 7(规则声明+测试)|
| 3.4 fail-open | Task 3/4/5(各自含 fail-open 测试)|
| 4.1 限流器 | Task 3 |
| 4.2 熔断器 | Task 4(Lua→SET NX 取舍已在任务头说明)|
| 4.3 超时 | Task 7(LLM)/ Task 8(embedding/rerank 配置化)|
| 4.4 降级 | Task 1(异常/文案)/ Task 8(rerank 降级)/ Task 9(SSE 友好报错)|
| 4.5 配置 | Task 2 / Task 9(.env.example)|
| 5.1-5.2 统计表与写入 | Task 5 |
| 5.3 聚合 API | Task 10 |
| 5.4 前端统计页 | Task 11 |
| 6 压测 | Task 12/13 |
| 7 测试策略 | 各任务内嵌(fakeredis 单测);集成测试项——真 Redis 并发抢配额——如需可在 Task 13 后补,不阻塞主线 |
