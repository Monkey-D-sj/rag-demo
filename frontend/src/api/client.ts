import type { ChatEvent, DocumentItem, RetryResult } from "@/types";

const BASE = "/api";

// ── Chat (POST + SSE 流) ────────────────────────────

/** 调用 /chat/ 返回 SSE 异步迭代器，解码后逐条 yield ChatEvent */
export async function* streamChat(
  sessionId: string,
  query: string,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${BASE}/chat/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "chat request failed");
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

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
        // 忽略解析失败的行
      }
    }
  }
}

// ── Documents ───────────────────────────────────────

export async function uploadDocument(
  file: File,
  knowledgeBaseId: string = "default",
): Promise<{ document_id: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("knowledge_base_id", knowledgeBaseId);

  const res = await fetch(`${BASE}/documents/`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "upload failed");
  }
  return res.json();
}

export async function getDocumentStatus(
  documentId: string,
): Promise<DocumentItem> {
  const res = await fetch(`${BASE}/documents/${documentId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "fetch document failed");
  }
  return res.json();
}

export async function retryDocument(
  documentId: string,
): Promise<RetryResult> {
  const res = await fetch(`${BASE}/documents/${documentId}/retry`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "retry failed");
  }
  return res.json();
}

// ── Health ──────────────────────────────────────────

export async function getHealth(): Promise<{
  status: string;
  checks: Record<string, string>;
}> {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}
