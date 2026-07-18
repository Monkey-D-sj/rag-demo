# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 代码规范

- 禁止重复声明已有的变量

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
# pytest addopts in pyproject.toml already skips integration+eval markers by default
uv run pytest tests/ -v

# Run a single test file/function
uv run pytest tests/test_memory.py -v
uv run pytest tests/test_memory.py::test_recall_memory_dedup -v

# Run integration tests (requires docker compose infrastructure)
uv run pytest tests/ -v -m integration

# Run retrieval eval (requires pg + embedding + seeded eval KB)
uv run pytest tests/ -v -m eval

# Start infrastructure only
docker compose up -d postgres redis minio neo4j

# Full stack (API + Worker + all infra)
docker compose up -d --build

# Frontend
cd frontend && pnpm install && pnpm dev
```

`pyproject.toml` defines three console scripts:
- `rag-api` (→ `rag.__main__:main`)
- `rag-worker` (→ `rag.worker.main:run`)
- `rag-eval` (→ `rag.eval.run:main`) — retrieval eval CLI with `--update-baseline` / `--classify` flags

## Architecture

### Dependency injection: lifespan → app.state → LangGraph Runtime

Resources (pg pool, redis, embedding model, LLM, memory manager, retriever, minio, neo4j, arq) are created in FastAPI `lifespan` via `AsyncExitStack` (LIFO rollback on startup failure), stored on `app.state`, then injected into LangGraph nodes through `Runtime[ContextSchema]`. The `ContextSchema` dataclass carries `llm`, `memory_manager`, `retriever`, and `reranker` — each node accesses them via `runtime.context`. The `MemoryManagerProtocol`, `RetrieverProtocol`, and `RerankerProtocol` in `rag/agent/type.py` define the interfaces; concrete implementations satisfy them structurally via `@runtime_checkable`.

The worker has its own independent startup (`on_startup`) that creates separate pg/minio/embedding/neo4j/llm instances — worker and API do not share connections.

### SSE streaming: producer-consumer pattern

`ChatStream` (`rag/api/common/stream.py`) is an `asyncio.Queue`-backed async iterable. A background `asyncio.Task` feeds LangGraph custom events (status/message/error) into the queue; the main coroutine yields SSE-formatted lines to FastAPI's `StreamingResponse`. `close()` enqueues `[DONE]` + a sentinel (`None`) to terminate iteration. The async context manager (`async with`) auto-closes on exit and emits errors.

### Agent workflow: 14-node pipeline with conditional routing

The state graph (`rag/agent/workflow.py`) is a 14-node pipeline with conditional branching:

```
                                     ┌─ out-of-scope → direct_answer ──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
START → recall_memory → handle_query ┤                                                                                                                                            END
                                     └─ in-scope → cache_lookup ┬─ hit → add_memory ───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                └─ miss → [Send × N] recall → recall_fuse → neighbor_expand → rerank → dynamic_topk → parent_expand ┬─ generate → cache_store → add_memory ─┘
                                                                                                                                                                    └─ no_results ──────────────────────────┘
```

| Node | Module | Role |
|------|--------|------|
| `recall_memory` | `nodes/recall_memory/` | Short-term (Redis) + Long-term (pgvector) memory recall via `MemoryManagerProtocol` |
| `handle_query` | `nodes/query/` | LLM structured output: scope check + query rewrite (指代消解 + 省略补全) + entity extraction for the graph recall leg + query decomposition (`sub_queries`, ≤3, gated by `QUERY_DECOMPOSITION_ENABLED` at routing). Outputs `is_out_of_scope`, `rewrite_query`, and `query_entities` |
| `cache_lookup` | `nodes/cache_lookup/` | Semantic cache lookup (pgvector similarity, global scope); on hit, streams the cached answer/citations and sets `cache_hit=True` to skip retrieval and generation (gated by `SEMANTIC_CACHE_ENABLED`; pass-through if disabled or not injected) |
| `recall` | `nodes/recall/` | Hybrid retrieval per Send branch: vector + BM25 + graph (main branch only) concurrent recall; branches carry `{sub_query, entities}` payloads |
| `recall_fuse` | `nodes/recall_fuse/` | Fan-in of Send branches: second-level RRF fusion across sub-query result lists (shared `fuse_multi_query_results()`), writes `recall_vec_results` |
| `neighbor_expand` | `nodes/neighbor_expand/` | Sentence Window: fetch ±N adjacent chunks from the same document (gated by `SENTENCE_WINDOW_ENABLED`; pass-through when disabled or pool missing) |
| `rerank` | `nodes/rerank/` | Semantic re-ranking via Qwen3-Rerank (optional; pass-through if no reranker injected) |
| `dynamic_topk` | `nodes/dynamic_topk/` | Adjacent-score-gap dynamic truncation of reranked results (with plateau protection) |
| `parent_expand` | `nodes/parent_expand/` | Parent-Child Retrieval: expand regulation-KB chunks to full parent document text by doc_id (gated by `PARENT_CHILD_ENABLED`; degrades to dedup-only on failure) |
| `generate` | `nodes/generate/` | RAG generation with retrieved context, SSE token streaming |
| `direct_answer` | `nodes/generate/` | Out-of-scope direct LLM answer (no retrieval, no memory write-back) |
| `no_results` | `nodes/generate/` | Static fallback message when recall is empty (no LLM call) |
| `cache_store` | `nodes/cache_store/` | Best-effort write-back of the generated answer + citations to the semantic cache (gated by `SEMANTIC_CACHE_ENABLED`; pass-through if disabled or not injected) |
| `add_memory` | `nodes/add_memory/` | Persists turn to short-term + long-term memory (best-effort, failures silently ignored) |

**Conditional routing:**
- `_route_after_query`: `is_out_of_scope=True` → `direct_answer`; otherwise → `cache_lookup`
- `_route_after_cache`: `cache_hit=True` → `add_memory`; otherwise returns `list[Send]` fanning out to `recall` — branches are `[rewrite_query] + sub_queries` (sub-queries only when `QUERY_DECOMPOSITION_ENABLED`), entities ride only on the main branch
- `_route_after_topk` (evaluated after `parent_expand`): `recall_vec_results` non-empty → `generate`; empty → `no_results`
- The `generate` path writes memory via `generate → cache_store → add_memory`; a cache hit writes memory directly via `cache_lookup → add_memory`; `direct_answer` and `no_results` go straight to `END`

**Key design patterns:**
- **Graceful degradation**: Memory recall failure → empty context. Retriever failure → empty results. Structured output failure → raw query passthrough. Reranker missing → pass-through.
- **Custom streaming only**: `stream_mode=["custom"]` — only `writer()` calls produce output events. State deltas are internal-only.
- **Dependency injection**: Resources (pg pool, redis, embedding, LLM, memory manager, retriever, reranker) are created in FastAPI `lifespan`, stored on `app.state`, injected via `Runtime[ContextSchema]`.

### State type: TypedDict with conditional routing fields

`MyState` (`rag/agent/type.py`) is a plain `TypedDict`. `sub_recall_results` is the only Annotated reducer field (`operator.add`, Send fan-in); all other fields are plain last-write-wins. Key fields:
- `session_id`, `raw_query`, `context` — input fields
- `is_out_of_scope` — set by `handle_query`; determines routing to `direct_answer` vs `cache_lookup`
- `rewrite_query` — rewritten query from `handle_query`; falls back to `raw_query` on structured-output failure
- `query_entities` — entities extracted by `handle_query`; code guarantees `[]` on structured-output failure, while an empty list on out-of-scope classification relies on prompt compliance rather than a code-level guard (that path never reaches `recall`, so it has no practical effect either way); passed to `recall` as the graph leg's seed entities
- `sub_queries` — sub-queries decomposed by `handle_query` (≤3, code-clamped); falls back to `[]` on structured-output failure; consumed by `_route_after_cache` to fan out Send branches only when `QUERY_DECOMPOSITION_ENABLED`
- `cache_hit` — set by `cache_lookup` on a semantic cache hit; `_route_after_cache` checks it to route straight to `add_memory`, skipping retrieval and generation
- `sub_recall_results` — Send fan-in reducer field; each `recall` branch appends a single-element list, `operator.add` concatenates across parallel branches; consumed by `recall_fuse`
- `recall_bm25_results`, `recall_vec_results` — `recall_vec_results` is written by `recall_fuse` (second-level RRF fusion of `sub_recall_results`), then transformed in place by `neighbor_expand` → `rerank` → `dynamic_topk` → `parent_expand` and used by `generate`; `_route_after_topk` checks `recall_vec_results` emptiness
- `generated` — final LLM response, persisted by `add_memory`

### Prompt management: centralized + structured output

All prompts live in `rag/prompts/` as module-level string constants. The `query.py` prompt handles two tasks in a single LLM call: scope judgment + query rewrite (指代消解 + 省略补全). Structured output uses Pydantic models (e.g., `QueryRewriteOutput`) with `ainvoke_structured()`.

The `generate.py` prompt includes context-aware RAG instructions. `rerank.py` and `eval.py` prompts are used by the reranker and golden-data generation respectively.

### LLM calling pattern

`NormalModel` wraps `ChatOpenAI` with tenacity `AsyncRetrying` (3 attempts, exponential jitter). The `is_retryable()` predicate in `rag/common/exception.py` distinguishes retryable (5xx, network errors) from non-retryable (4xx, auth, rate limit) failures. HTTP status codes from provider errors are mapped to typed exceptions via `from_http_error()`.

Structured output uses `json_mode` (not function calling) with explicit JSON Schema injected into the system message — this avoids provider-specific limitations (e.g., DeepSeek thinking mode rejecting `tool_choice`). `OutputParserException` and `ValidationError` are also treated as retryable.

Streaming (`astream`) does NOT retry — retry semantics for mid-stream failures are intentionally deferred.

### Hybrid retrieval: vector + BM25 + graph with RRF fusion

`KnowledgeRetriever.search(query, knowledge_base_ids, top_k, query_emb=None, entities=None)` fires three concurrent legs via `asyncio.gather`: vector recall (pgvector cosine distance), BM25 recall (ParadeDB `pg_search` with jieba tokenizer), and graph recall (1-hop expansion over `entities` via `GraphRetriever`, only when a graph retriever is injected and `entities` is non-empty). All three legs degrade independently to empty results on failure — vector is the primary quality source in practice, not the only leg that stays up: BM25 and graph failures are caught the same way vector failures are, and the graph leg additionally bounds itself with `asyncio.wait_for(..., timeout=GRAPH_RECALL_TIMEOUT_SECONDS)`, so a slow Neo4j query degrades to a two-leg result instead of blocking the whole search. Each leg fetches `top_k * candidate_multiplier` candidates, then `_merge_dedup()` fuses them via Reciprocal Rank Fusion (RRF, constant `RETRIEVER_RRF_K`): per-leg RRF scores are summed for chunks hit by multiple legs, marking `sources` on each row (`["vec"]`, `["bm25"]`, `["graph"]`, or any combination). Final ranking is delegated to the reranker node.

Lexical queries are sanitized (punctuation stripped) before BM25 to avoid noise. Vector results below `RETRIEVER_VEC_SIMILARITY_THRESHOLD` are filtered before merging. `KnowledgeRetriever.has_graph` reports whether a graph retriever is injected (used by eval's `graph_fused` leg to decide whether to run).

### Embedding: batch size limit

The embedding API (DashScope/Alibaba compatible mode) enforces a **maximum of 10 texts per batch** (`InternalError.Algo.InvalidParameter` if exceeded). The default `EMBEDDING_BATCH_SIZE` is 10 — do not raise it without verifying the target API's limit. The embedding model is `text-embedding-v4` with 1024 dimensions.

### Logging: custom get_logger + contextvars for session propagation

Always use `from rag.common.logging import get_logger` instead of `logging.getLogger(__name__)` — it uses stack inspection to derive the caller's module name, so copy-pasting the one-liner works correctly. Session ID propagation uses `contextvars` (`_session_id_var`): `bind_session(session_id)` sets it for the request's async context, `_SessionContextFilter` attaches it to every log record. `setup_logging()` is idempotent and takes over uvicorn/uvicorn.access/uvicorn.error/watchfiles loggers (clears their handlers, sets `propagate=True`). Log format is controlled by `LOG_FORMAT` (`text` with optional color, or `json`). Loki push runs on a background daemon thread with batching.

### Observability: null-object pattern for zero overhead when disabled

All Langfuse integration in `rag/observability/langfuse.py` follows the same pattern: when `LANGFUSE_ENABLED=false` or keys are missing, functions return passthrough/noop versions. `observe_if_enabled`/`observe_root` return the original function unchanged; `get_callback_handler` returns `None`; `span_scope`/`session_scope` return `contextlib.nullcontext()`. Lazy imports (`from langfuse import ...`) ensure the SDK is never loaded when disabled. The `@lru_cache` on `_init_client()` ensures the global Langfuse client is created at most once per process.

The same `*_ENABLED` toggle pattern applies to **Loki** (`LOKI_ENABLED`), **Neo4j** (`NEO4J_ENABLED`), and **entity extraction** (`ENABLE_ENTITY_EXTRACTION`) — all default to `false` and require explicit opt-in.

### Document pipeline: state machine + self-healing

Documents flow through `pending → processing → done|failed`. `claim_for_processing()` atomically transitions `pending`/`failed` → `processing` (defense against concurrent workers). Worker cron runs every 5 minutes:
- **failed**: retry with exponential backoff (`backoff_base * 2^retry_count` seconds), up to `MAX_RETRY_ROUNDS`
- **stalled**: `pending`/`processing` documents stuck longer than `STALE_DOC_SECONDS` are reclaimed

Entity extraction (graph pipeline) is a separate ARQ task (`extract_document_entities`, timeout=900s). It uses `BoundedSemaphore` to limit concurrent LLM calls. Single-chunk failures are skipped; all-chunk failures mark `graph_status=failed` for cron retry. `asyncio.CancelledError` (from ARQ timeout) is caught separately to set `failed` status before re-raising.

### Database: async psycopg3 with dict_row cursors

All DB access goes through `get_cursor(pool)` — an async context manager yielding dict-row cursors. Each connection auto-registers pgvector via `_configure`. Transactions auto-commit on clean exit and rollback on exception. There is no ORM; all queries are raw SQL. Alembic migrations live in `alembic/versions/` and use the sync `PG_SYNC_URL` (note: migrations use `psycopg` sync driver, not the async pool).

### Retrieval eval system: 8-leg breakdown + gate + history

`rag/eval/` provides a retrieval quality regression suite:
- **Golden dataset**: `rag/eval/datasets/retrieval_golden.jsonl` — hand-curated query → relevant doc_ids pairs (西游记 themed); items may also carry an optional `entities` annotation consumed by the `graph_fused` leg
- **8 evaluation legs**: `fused` (混合改写), `fused_reranked` (混合重排), `raw` (混合原文), `vec_only` (向量改写), `raw_vec` (向量原文), `bm25_only` (BM25改写), `raw_bm25` (BM25原文), `graph_fused` (三路融合: 向量+BM25+图) — the `raw*` legs only run when golden items carry `rewrite_query`; `fused_reranked` only when a reranker is injected; `graph_fused` only when `retriever.has_graph` is true and the golden item carries `entities`
- **Metrics**: hit@k, recall@k, ndcg@k, mrr — computed per-query, then aggregated
- **History**: each run saves `history/YYYYMMDD-HHMMSS-{commit}.json` with per-leg aggregate + per-query breakdown
- **Gate**: `gate()` compares aggregate metrics against `baseline.json` thresholds; fails on >3% relative drop in recall@5 or mrr. `rewrite_gate()` separately checks that query rewriting doesn't degrade retrieval
- **Classify mode** (`--classify`): runs LLM `handle_query` classification against golden `out_of_scope` labels, computes Precision/Recall/F1
- **Breakdown table**: CJK-aligned multi-column terminal output showing all legs side-by-side, plus rewrite gain and rerank gain summaries
- **Seed corpus**: `rag/eval/seed_corpus.py` populates an isolated eval environment from source documents
- **Golden generation**: `rag/eval/generate_golden.py` uses LLM structured output to produce candidate query-document pairs for manual curation
- **CLI**: `uv run rag-eval [--update-baseline] [--classify]` or `uv run pytest tests/ -v -m eval`

The eval environment is deliberately isolated from production. The eval retriever uses `EVAL_KB_ID` for knowledge-base-based routing.

### Exception handling in API

Service layer only raises `AppError` subclasses (from `rag/common/exception.py`). The `register_error_handlers` function in `rag/api/common/error_handlers.py` maps `AppError.status_code` → HTTP response. LLM exceptions (`LLMException` hierarchy) are separate and handled at the model layer with retry logic.

### Frontend: React 18 + Vite + TailwindCSS

`frontend/` is a single-page app with three pages (`ChatPage`, `DocumentsPage`, `HealthPage`). SSE streaming consumption lives in `frontend/src/api/stream.ts` — it reads the same custom events emitted by the LangGraph workflow. API calls go through `frontend/src/api/client.ts` (thin wrapper around `fetch`). Components are plain React — no state management library beyond React hooks.

### Windows compatibility

`setup_windows_loop()` in `rag/common/platform.py` sets `asyncio.WindowsSelectorEventLoopPolicy()` when `sys.platform == "win32"`. This is called at the top of: `rag/__main__.py` (API entry), `rag/api/main.py` (ASGI import-time for reload workers), `rag/worker/main.py` (worker entry), `rag/eval/run.py` (eval CLI), and `tests/conftest.py`. This is needed because psycopg3 async requires `SelectorEventLoop`, but Windows defaults to `ProactorEventLoop`. Uvicorn is started with `loop="none"` to prevent it from overriding the policy.
