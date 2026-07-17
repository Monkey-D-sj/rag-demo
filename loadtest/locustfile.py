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
