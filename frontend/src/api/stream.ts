import type { ChatEvent } from "@/types";

const BASE = "/api";

export interface ChatStream {
  [Symbol.asyncIterator](): AsyncIterator<ChatEvent>;
  abort(): void;
}

// 创建 Chat SSE 流——封装 fetch + ReadableStream 解析 + SSE 反序列化。
// 用法: for await (const ev of createChatStream(sid, q)) { ... }
export function createChatStream(
  sessionId: string,
  query: string,
): ChatStream {
  const controller = new AbortController();

  const iterator = (async function* (): AsyncIterator<ChatEvent> {
    const res = await fetch(`${BASE}/chat/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, query }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? "chat request failed");
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          const payload = line.slice(6);

          if (payload === "[DONE]") {
            yield { type: "done", data: null };
            return;
          }

          try {
            yield JSON.parse(payload) as ChatEvent;
          } catch {
            // 解析失败的行静默跳过
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  })();

  return {
    [Symbol.asyncIterator]: () => iterator,
    abort: () => controller.abort(),
  };
}
