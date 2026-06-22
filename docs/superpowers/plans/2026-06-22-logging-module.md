# 日志记录模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为项目提供中央日志配置 `rag/common/logging.py`,统一 level、格式(text/json 可切换)、输出目标(控制台 + 可选轮转文件)。

**Architecture:** 一个幂等的 `setup_logging()` 配置 root logger,各业务模块继续用 `logging.getLogger(__name__)` 不改动。格式由 `JsonFormatter` 与彩色文本 formatter 提供,配置走现有 pydantic `Settings`。

**Tech Stack:** Python 标准库 `logging`(`StreamHandler`/`RotatingFileHandler`/`Formatter`)、`json`;pydantic-settings;pytest。

## Global Constraints

- **不引入任何新依赖**——仅用标准库实现 JSON/文本格式与文件轮转。
- 注释/文档用中文,与现有代码风格一致。
- 业务模块保持 `logging.getLogger(__name__)`,本期不改动它们。
- 配置字段加到现有 `rag/config.py` 的 `Settings`,可由 `.env` 覆盖。
- 测试每个用例前后清理 root logger 的 handler,避免污染其他测试。
- 提交信息用中文 `feat:` / `test:` 前缀,与近期提交风格一致。

---

### Task 1: 在 Settings 中新增 log 配置字段

**Files:**
- Modify: `rag/config.py`(在 `# ── Embedding ──` 块之后、`@property` 之前插入)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 现有 `Settings`(pydantic `BaseSettings`)。
- Produces: `Settings` 新增属性 `log_level: str`、`log_format: str`、`log_file: str | None`、`log_file_max_bytes: int`、`log_file_backup_count: int`,供 Task 2~4 使用。

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 末尾追加:

```python
def test_settings_has_log_defaults():
    from rag.config import Settings

    s = Settings()
    assert s.log_level == "INFO"
    assert s.log_format == "text"
    assert s.log_file is None
    assert s.log_file_max_bytes == 10 * 1024 * 1024
    assert s.log_file_backup_count == 5
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_config.py::test_settings_has_log_defaults -v`
Expected: FAIL,`AttributeError: 'Settings' object has no attribute 'log_level'`

- [ ] **Step 3: 实现最小代码**

在 `rag/config.py` 的 `embedding_dim: int = 1024` 这一行之后、`@property def pg_async_dsn` 之前,插入:

```python

    # ── Logging ──
    log_level: str = "INFO"                        # DEBUG / INFO / WARNING / ERROR
    log_format: str = "text"                       # "text"（开发） | "json"（生产）
    log_file: str | None = None                    # None=仅控制台；给路径则额外写文件
    log_file_max_bytes: int = 10 * 1024 * 1024     # 单文件 10MB
    log_file_backup_count: int = 5                 # 轮转保留份数
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_config.py::test_settings_has_log_defaults -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/config.py tests/test_config.py
git commit -m "feat: Settings 新增 log_* 配置字段"
```

---

### Task 2: JsonFormatter

**Files:**
- Create: `rag/common/logging.py`
- Test: `tests/test_logging.py`(新建)

**Interfaces:**
- Consumes: 标准库 `logging`、`json`。
- Produces: `class JsonFormatter(logging.Formatter)`,其 `format(record)` 返回单行 JSON 字符串,含 `timestamp`(ISO8601)、`level`、`logger`、`message`;record 含异常时附 `exc_info`(格式化堆栈字符串)。Task 4 的 `setup_logging` 会用到它。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_logging.py`:

```python
import json
import logging

from rag.common.logging import JsonFormatter


def _record(level=logging.INFO, msg="hello", exc_info=None):
    return logging.LogRecord(
        name="rag.test", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )


def test_json_formatter_basic_fields():
    out = JsonFormatter().format(_record(msg="hello world"))
    data = json.loads(out)
    assert data["level"] == "INFO"
    assert data["logger"] == "rag.test"
    assert data["message"] == "hello world"
    assert "timestamp" in data


def test_json_formatter_includes_exc_info():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = _record(level=logging.ERROR, msg="failed", exc_info=sys.exc_info())
    data = json.loads(JsonFormatter().format(rec))
    assert "exc_info" in data
    assert "ValueError: boom" in data["exc_info"]
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_logging.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'rag.common.logging'`

- [ ] **Step 3: 实现最小代码**

新建 `rag/common/logging.py`:

```python
import datetime
import json
import logging


class JsonFormatter(logging.Formatter):
    """每行一个 JSON 的日志格式器（纯标准库）。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(record.created).isoformat()
        data: dict = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_logging.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add rag/common/logging.py tests/test_logging.py
git commit -m "feat: 日志 JsonFormatter"
```

---

### Task 3: 彩色文本 formatter

**Files:**
- Modify: `rag/common/logging.py`
- Test: `tests/test_logging.py`

**Interfaces:**
- Consumes: 标准库 `logging`。
- Produces: `class ColorTextFormatter(logging.Formatter)`,构造参数 `use_color: bool = True`;`use_color=True` 时给 `levelname` 加 ANSI 颜色,否则纯文本。基础格式 `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`,`datefmt="%Y-%m-%d %H:%M:%S"`。Task 4 用到。

- [ ] **Step 1: 写失败测试**

在 `tests/test_logging.py` 末尾追加:

```python
from rag.common.logging import ColorTextFormatter


def test_text_formatter_plain_has_no_ansi():
    out = ColorTextFormatter(use_color=False).format(_record(msg="hi"))
    assert "\x1b[" not in out
    assert "INFO" in out
    assert "rag.test" in out
    assert "hi" in out


def test_text_formatter_color_has_ansi():
    out = ColorTextFormatter(use_color=True).format(_record(msg="hi"))
    assert "\x1b[" in out
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_logging.py -k text_formatter -v`
Expected: FAIL,`ImportError: cannot import name 'ColorTextFormatter'`

- [ ] **Step 3: 实现最小代码**

在 `rag/common/logging.py` 顶部 `JsonFormatter` 之后追加:

```python
_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",     # cyan
    "INFO": "\x1b[32m",      # green
    "WARNING": "\x1b[33m",   # yellow
    "ERROR": "\x1b[31m",     # red
    "CRITICAL": "\x1b[1;31m",
}
_RESET = "\x1b[0m"

_TEXT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class ColorTextFormatter(logging.Formatter):
    """可读文本格式器；use_color=True 时给 level 上 ANSI 颜色。"""

    def __init__(self, use_color: bool = True):
        super().__init__(fmt=_TEXT_FMT, datefmt=_DATE_FMT)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelname, "")
            if color:
                # 复制以免污染原始 record（影响其他 handler）
                record = logging.makeLogRecord(record.__dict__)
                record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_logging.py -k text_formatter -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add rag/common/logging.py tests/test_logging.py
git commit -m "feat: 日志彩色文本 formatter"
```

---

### Task 4: setup_logging

**Files:**
- Modify: `rag/common/logging.py`
- Test: `tests/test_logging.py`

**Interfaces:**
- Consumes: `JsonFormatter`、`ColorTextFormatter`(Task 2/3);`Settings`、`get_settings`(`rag/config.py`,Task 1)。
- Produces: `def setup_logging(settings: Settings | None = None) -> None`——幂等配置 root logger:清空已有 handler,按 `log_format` 选 formatter,加 stdout `StreamHandler`,`log_file` 非空时加 `RotatingFileHandler`(`maxBytes`/`backupCount`/`encoding="utf-8"`,先 `mkdir` 父目录),设置 level(非法 level 回退 `INFO` 并 WARNING),收编 `uvicorn`/`uvicorn.access`/`uvicorn.error`(清 handler、`propagate=True`)。

- [ ] **Step 1: 写失败测试**

在 `tests/test_logging.py` 末尾追加(含清理 fixture):

```python
import json as _json
import pytest

from rag.common.logging import setup_logging
from rag.config import Settings


@pytest.fixture(autouse=True)
def _clean_root_handlers():
    root = logging.getLogger()
    saved = root.handlers[:]
    saved_level = root.level
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(saved)
    root.setLevel(saved_level)


def test_setup_logging_text_adds_stream_handler():
    setup_logging(Settings(log_format="text"))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_setup_logging_is_idempotent():
    s = Settings(log_format="text")
    setup_logging(s)
    setup_logging(s)
    assert len(logging.getLogger().handlers) == 1


def test_setup_logging_respects_level():
    setup_logging(Settings(log_level="WARNING"))
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_json_output_is_valid_json(capsys):
    setup_logging(Settings(log_format="json"))
    logging.getLogger("rag.test").error("boom")
    err = capsys.readouterr().err
    line = [ln for ln in err.splitlines() if ln.strip()][-1]
    data = _json.loads(line)
    assert data["message"] == "boom"
    assert data["level"] == "ERROR"


def test_setup_logging_creates_file_and_parent_dir(tmp_path):
    log_path = tmp_path / "logs" / "app.log"
    setup_logging(Settings(log_file=str(log_path)))
    from logging.handlers import RotatingFileHandler
    root = logging.getLogger()
    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)
    logging.getLogger("rag.test").error("written")
    for h in root.handlers:
        h.flush()
    assert log_path.exists()
    assert "written" in log_path.read_text(encoding="utf-8")


def test_setup_logging_invalid_level_falls_back_to_info():
    setup_logging(Settings(log_level="NOTALEVEL"))
    assert logging.getLogger().level == logging.INFO
```

> 注:JSON handler 默认指向 `sys.stderr`(`StreamHandler()` 默认 stream),故用 `capsys.readouterr().err`。

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_logging.py -k setup_logging -v`
Expected: FAIL,`ImportError: cannot import name 'setup_logging'`

- [ ] **Step 3: 实现最小代码**

在 `rag/common/logging.py` 顶部补充 import,并在文件末尾追加 `setup_logging`:

```python
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rag.config import Settings, get_settings

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _build_formatter(log_format: str, *, use_color: bool) -> logging.Formatter:
    if log_format == "json":
        return JsonFormatter()
    return ColorTextFormatter(use_color=use_color)


def setup_logging(settings: Settings | None = None) -> None:
    """幂等配置 root logger。可在应用启动时重复调用。"""
    settings = settings or get_settings()

    root = logging.getLogger()
    root.handlers.clear()

    # 解析 level（非法则回退 INFO）
    level_name = settings.log_level.upper()
    invalid_level = level_name not in _VALID_LEVELS
    level = logging.INFO if invalid_level else getattr(logging, level_name)
    root.setLevel(level)

    # 控制台 handler（默认 stderr）
    stream_handler = logging.StreamHandler()
    use_color = bool(getattr(stream_handler.stream, "isatty", lambda: False)())
    stream_handler.setFormatter(
        _build_formatter(settings.log_format, use_color=use_color)
    )
    root.addHandler(stream_handler)

    # 可选文件 handler（始终非彩色）
    if settings.log_file:
        path = Path(settings.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            _build_formatter(settings.log_format, use_color=False)
        )
        root.addHandler(file_handler)

    # 收编 uvicorn，统一走 root
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    if invalid_level:
        logging.getLogger(__name__).warning(
            "未知 log_level %r，已回退为 INFO", settings.log_level
        )
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_logging.py -v`
Expected: PASS(全部通过)

- [ ] **Step 5: 提交**

```bash
git add rag/common/logging.py tests/test_logging.py
git commit -m "feat: setup_logging 中央日志配置"
```

---

### Task 5: 在入口接入 setup_logging

**Files:**
- Modify: `main.py`
- Modify: `rag/api/main.py`

**Interfaces:**
- Consumes: `setup_logging`(Task 4)。
- Produces: 无新接口;仅在两个入口启动时调用 `setup_logging()`。

> 注:`rag/api/main.py` 当前不完整(`import FastAPI` 写法有误、`lifespan` 体未收尾)。本任务**只新增** `setup_logging()` 调用,不修复其它既有问题(超出本计划 scope)。

- [ ] **Step 1: 修改 `main.py`**

将 `main.py` 内容替换为:

```python
import logging

from rag.common.logging import setup_logging


def main() -> None:
    setup_logging()
    logging.getLogger(__name__).info("rag-demo 启动")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在 `rag/api/main.py` 的 lifespan 中调用**

在 `rag/api/main.py` 的 `lifespan` 函数体开头(`pool = await create_pg_pool()` 之前)加入:

```python
    from rag.common.logging import setup_logging
    setup_logging()
```

- [ ] **Step 3: 冒烟验证 main.py**

Run: `python main.py`
Expected: 输出一行形如 `2026-06-22 10:00:00 | INFO | __main__ | rag-demo 启动`(无 traceback)

- [ ] **Step 4: 跑全量测试确认无回归**

Run: `pytest -q`
Expected: 全部通过(不依赖外部 DB 的用例)。若有需要 DB/网络的用例失败,确认与本改动无关。

- [ ] **Step 5: 提交**

```bash
git add main.py rag/api/main.py
git commit -m "feat: 入口接入 setup_logging"
```

---

## Self-Review

**1. Spec coverage:**
- 中央配置模块 `rag/common/logging.py` → Task 2/3/4 ✅
- Settings 5 个 `log_*` 字段 → Task 1 ✅
- text/json 切换 → Task 2(json)、Task 3(text)、Task 4(选择逻辑)✅
- 控制台 + 可选轮转文件 → Task 4 ✅
- 目录自动创建 → Task 4 + 测试 ✅
- 幂等 → Task 4 + 测试 ✅
- 非法 level 回退 → Task 4 + 测试 ✅
- 收编 uvicorn → Task 4 ✅
- 接入 main.py / lifespan → Task 5 ✅
- 测试覆盖(格式/level/文件/幂等/目录)→ Task 2/3/4 测试 ✅
- 不引入新依赖 → 全程仅标准库 ✅

**2. Placeholder scan:** 无 TBD/TODO,每个代码步骤含完整代码。✅

**3. Type consistency:** `JsonFormatter`、`ColorTextFormatter(use_color=...)`、`setup_logging(settings=None)`、`_build_formatter(log_format, use_color=...)` 在各 Task 间命名一致。✅

> 说明:彩色文本测试(Task 3)直接构造 `ColorTextFormatter(use_color=True)`,不依赖 TTY,稳定;`setup_logging` 中颜色由 `stream.isatty()` 决定,测试环境下通常为非彩色,与断言不冲突。
