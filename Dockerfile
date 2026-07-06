# syntax=docker/dockerfile:1
# 生产镜像:api 与 worker 共用同一镜像,靠 command 区分角色(rag-api / rag-worker)。
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先只装依赖(锁文件不变则命中缓存,改源码不必重装依赖)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 再拷源码并安装项目本身(提供 rag-api / rag-worker console scripts)
COPY rag ./rag
COPY alembic ./alembic
COPY alembic.ini main.py README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# venv 可执行入 PATH
ENV PATH="/app/.venv/bin:$PATH"

# 非 root 运行
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 默认以 API 启动;worker 在 compose 里用 command 覆盖为 rag-worker
CMD ["rag-api"]
