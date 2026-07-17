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
