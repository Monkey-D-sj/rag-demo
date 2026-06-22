# 日志记录模块设计

日期:2026-06-22
状态:已认可,待实现

## 目标

为项目提供**中央日志配置**。当前项目已在 `rag/models/embedding.py`、`rag/models/normal.py`
中使用标准库 `logging.getLogger(__name__)` 模式,并配合 tenacity 的 `before_sleep_log`,
但**没有任何中央配置**——没人调用过 `logging.basicConfig` 或配置 handler/formatter,
导致这些日志要么不输出、要么走 Python 默认的 WARNING 级裸输出。

本模块统一 level、格式、输出目标,各业务模块保持现状(继续用 `getLogger(__name__)`),无需改动。

## 非目标(YAGNI)

明确排除以下内容,以后需要再加:

- **请求/会话级关联 ID**(`request_id` / `session_id` via `contextvars`)——价值高但增加复杂度,本期不做。
- **异步 `QueueHandler`**——当前吞吐量用不上。
- **按时间轮转**(`TimedRotatingFileHandler`)——按大小轮转已足够。

## 架构

新增模块 **`rag/common/logging.py`**,提供幂等的 `setup_logging()` 函数,在应用启动时调用一次。

- 统一配置 **root logger**,使 `rag.*` 与第三方库(uvicorn 等)的日志都走同一套格式/输出。
- 业务模块保持 `logging.getLogger(__name__)` 不变。

## 配置

在现有 `rag/config.py` 的 `Settings` 中新增以下字段(均可由 `.env` 覆盖,与现有 pydantic 配置风格一致):

```python
log_level: str = "INFO"                        # DEBUG / INFO / WARNING / ERROR
log_format: str = "text"                       # "text"(开发) | "json"(生产)
log_file: str | None = None                    # None=仅控制台;给路径则额外写文件
log_file_max_bytes: int = 10 * 1024 * 1024     # 单文件 10MB
log_file_backup_count: int = 5                 # 轮转保留份数
```

## 组件

### `setup_logging(settings: Settings | None = None) -> None`

- `settings` 为 `None` 时调用 `get_settings()` 获取。
- **幂等**:先清空 root logger 已有的 handler(`logger.handlers.clear()`),保证可重复调用不叠加。
- 按 `log_format` 选择 formatter,装配 handler,设置 `log_level`。
- 始终添加一个指向 `stderr`（`StreamHandler` 默认 stream）的 `StreamHandler`。
- 当 `log_file` 非空时,额外添加一个 `RotatingFileHandler`
  (`maxBytes=log_file_max_bytes`,`backupCount=log_file_backup_count`,`encoding="utf-8"`)。
  文件 handler 始终使用非彩色 formatter。
- 收编 uvicorn:将 `uvicorn`、`uvicorn.access`、`uvicorn.error` logger 的 `handlers` 清空、
  `propagate=True`,使其日志通过 root 统一输出,避免双份/不一致。

### `JsonFormatter(logging.Formatter)`

- 每行输出一个 JSON 对象(`json.dumps(..., ensure_ascii=False)`)。
- 字段:`timestamp`(ISO8601)、`level`、`logger`(record.name)、`message`。
- 当 record 含异常时,附带 `exc_info`(格式化后的堆栈字符串)。
- 纯标准库 `json`,**不引入额外依赖**。

### 文本 formatter

- 格式:`%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`,
  `asctime` 形如 `2026-06-22 10:00:00`。
- 仅当输出为 TTY(`stream.isatty()`)时,给 `levelname` 加 ANSI 颜色;
  非 TTY 或文件输出时关闭颜色。通过一个 `ColorTextFormatter` 实现,文件 handler 用普通文本 formatter。

## 接入点

- `main.py`:在入口调用 `setup_logging()`。
- `rag/api/main.py`:在 FastAPI `lifespan` 开头调用 `setup_logging()`。
- 收编 uvicorn 的逻辑封装在 `setup_logging()` 内,接入点无需额外处理。

## 错误处理

- `log_file` 指向的目录不存在时,`setup_logging()` 负责 `mkdir(parents=True, exist_ok=True)` 创建父目录。
- `log_level` 非法时,回退到 `INFO` 并发一条 WARNING(用刚配好的 logger)。

## 测试(`tests/test_logging.py`)

1. **text 模式**:`setup_logging()` 后 root logger 含预期数量的 handler,且 formatter 类型正确。
2. **json 模式**:捕获一条日志输出,断言是合法 JSON 且含 `timestamp / level / logger / message` 字段;
   带异常时含 `exc_info`。
3. **level 生效**:设为 `WARNING` 时,`INFO`/`DEBUG` 记录被过滤。
4. **文件输出**:给 `log_file`(用 pytest `tmp_path`)时,`RotatingFileHandler` 被创建,写入后文件存在且含内容。
5. **幂等**:连续调用 `setup_logging()` 两次,root handler 数量不翻倍。
6. **目录自动创建**:`log_file` 指向不存在的子目录时,父目录被创建。

每个测试用例前后清理 root logger 的 handler,避免污染其他测试(用 fixture)。
