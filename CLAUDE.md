# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies
uv sync

# Run database migrations
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"  # create new migration

# Start API (hot reload in dev)
RAG_RELOAD=1 uv run rag-api
# or: python -m rag

# Start worker
uv run rag-worker

# Run tests (unit only, no external services)
uv run pytest tests/ -v --ignore=tests/test_db.py --ignore=tests/test_db_neo4j.py

# Run a single test file/function
uv run pytest tests/test_memory.py -v
uv run pytest tests/test_memory.py::test_recall_memory_dedup -v

# Run integration tests (requires docker compose infrastructure)
uv run pytest tests/ -v -m integration

# Start infrastructure only
docker compose up -d postgres redis minio neo4j

# Full stack (API + Worker + all infra)
docker compose up -d --build

# Frontend
cd frontend && pnpm install && pnpm dev
```

`pyproject.toml` defines two console scripts: `rag-api` (→ `rag.__main__:main`) and `rag-worker` (→ `rag.worker.main:run`).

## Architecture

### Dependency injection: lifespan → app.state → LangGraph Runtime

Resources (pg pool, redis, embedding model, LLM, memory manager, retriever, minio, neo4j, arq) are created in FastAPI `lifespan` via `AsyncExitStack` (LIFO rollback on startup failure), stored on `app.state`, then injected into LangGraph nodes through `Runtime[ContextSchema]`. The `ContextSchema` dataclass carries `llm`, `memory_manager`, and `retriever` — each node accesses them via `runtime.context`. The `MemoryManagerProtocol` and `RetrieverProtocol` in `rag/agent/type.py` define the interfaces; concrete implementations (`MemoryManager`, `KnowledgeRetriever`) satisfy them structurally via `@runtime_checkable`.

The worker has its own independent startup (`on_startup`) that creates separate pg/minio/embedding/neo4j/llm instances — worker and API do not share connections.

### SSE streaming: producer-consumer pattern

`ChatStream` (`rag/api/common/stream.py`) is an `asyncio.Queue`-backed async iterable. A background `asyncio.Task` feeds LangGraph custom events (status/message/error) into the queue; the main coroutine yields SSE-formatted lines to FastAPI's `StreamingResponse`. `close()` enqueues `[DONE]` + a sentinel (`None`) to terminate iteration. The async context manager (`async with`) auto-closes on exit and emits errors.

### LangGraph workflow: custom events only

The state graph (`rag/agent/workflow.py`) is a linear 4-node pipeline: `recall_memory → handle_query → recall → generate`. It uses `stream_mode=["custom"]` exclusively — the `updates` channel (state diffs) is disabled. Nodes communicate via `get_stream_writer()` emitting typed dicts: `{"type": "status"|"message"|"error", "data": "..."}`. The `handle_query` node (query rewrite) is defined but currently passes through unchanged.

After generation, `generate` node calls `_persist_turn()` to write both user query and assistant answer into short-term (Redis list, capped + TTL) and long-term (pgvector, per-session) memory. Memory write failures are logged but never propagate to the user.

### LLM calling pattern

`NormalModel` wraps `ChatOpenAI` with tenacity `AsyncRetrying` (3 attempts, exponential jitter). The `is_retryable()` predicate in `rag/common/exception.py` distinguishes retryable (5xx, network errors) from non-retryable (4xx, auth, rate limit) failures. HTTP status codes from provider errors are mapped to typed exceptions via `from_http_error()`.

Structured output uses `json_mode` (not function calling) with explicit JSON Schema injected into the system message — this avoids provider-specific limitations (e.g., DeepSeek thinking mode rejecting `tool_choice`). `OutputParserException` and `ValidationError` are also treated as retryable.

Streaming (`astream`) does NOT retry — retry semantics for mid-stream failures are intentionally deferred.

### Hybrid retrieval: vector + BM25 with RRF fusion

`KnowledgeRetriever.search()` fires vector recall (pgvector cosine distance) and BM25 recall (ParadeDB `pg_search` with jieba tokenizer) concurrently via `asyncio.gather`. BM25 failures are caught and degraded to empty results — vector is the critical path. Both legs fetch `top_k * 2` candidates, then RRF (Reciprocal Rank Fusion, k=60) merges and re-ranks to final `top_k`. Lexical queries are sanitized (punctuation stripped) before BM25 to avoid noise.

### Logging: custom get_logger + contextvars for session propagation

Always use `from rag.common.logging import get_logger` instead of `logging.getLogger(__name__)` — it uses stack inspection to derive the caller's module name, so copy-pasting the one-liner works correctly. Session ID propagation uses `contextvars` (`_session_id_var`): `bind_session(session_id)` sets it for the request's async context, `_SessionContextFilter` attaches it to every log record. `setup_logging()` is idempotent and takes over uvicorn/uvicorn.access/uvicorn.error/watchfiles loggers (clears their handlers, sets `propagate=True`). Log format is controlled by `LOG_FORMAT` (`text` with optional color, or `json`). Loki push runs on a background daemon thread with batching.

### Observability: null-object pattern for zero overhead when disabled

All Langfuse integration in `rag/observability/langfuse.py` follows the same pattern: when `LANGFUSE_ENABLED=false` or keys are missing, functions return passthrough/noop versions. `observe_if_enabled`/`observe_root` return the original function unchanged; `get_callback_handler` returns `None`; `span_scope`/`session_scope` return `contextlib.nullcontext()`. Lazy imports (`from langfuse import ...`) ensure the SDK is never loaded when disabled. The `@lru_cache` on `_init_client()` ensures the global Langfuse client is created at most once per process.

### Document pipeline: state machine + self-healing

Documents flow through `pending → processing → done|failed`. `claim_for_processing()` atomically transitions `pending`/`failed` → `processing` (defense against concurrent workers). Worker cron runs every 5 minutes:
- **failed**: retry with exponential backoff (`backoff_base * 2^retry_count` seconds), up to `MAX_RETRY_ROUNDS`
- **stalled**: `pending`/`processing` documents stuck longer than `STALE_DOC_SECONDS` are reclaimed

Entity extraction (graph pipeline) is a separate ARQ task (`extract_document_entities`, timeout=900s). It uses `BoundedSemaphore` to limit concurrent LLM calls. Single-chunk failures are skipped; all-chunk failures mark `graph_status=failed` for cron retry. `asyncio.CancelledError` (from ARQ timeout) is caught separately to set `failed` status before re-raising.

### Database: async psycopg3 with dict_row cursors

All DB access goes through `get_cursor(pool)` — an async context manager yielding dict-row cursors. Each connection auto-registers pgvector via `_configure`. Transactions auto-commit on clean exit and rollback on exception. There is no ORM; all queries are raw SQL. Alembic migrations live in `alembic/versions/` and use the sync `PG_SYNC_URL` (note: migrations use `psycopg` sync driver, not the async pool).

### Exception handling in API

Service layer only raises `AppError` subclasses (from `rag/common/exception.py`). The `register_error_handlers` function in `rag/api/common/error_handlers.py` maps `AppError.status_code` → HTTP response. LLM exceptions (`LLMException` hierarchy) are separate and handled at the model layer with retry logic.

### Windows compatibility

Three places set `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())`: `rag/__main__.py` (API entry), `rag/api/main.py` (ASGI import-time for reload workers), `rag/worker/main.py` (worker entry), and `tests/conftest.py`. This is needed because psycopg3 async requires `SelectorEventLoop`, but Windows defaults to `ProactorEventLoop`. Uvicorn is started with `loop="none"` to prevent it from overriding the policy.
