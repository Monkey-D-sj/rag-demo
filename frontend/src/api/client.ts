import type {
  DocumentItem,
  DocumentListResponse,
  DocumentStatus,
  GraphRetryResult,
  LlmSummaryResponse,
  RetryResult,
  Session,
} from "@/types";

const BASE = "/api";

// ── Documents ───────────────────────────────────────

export async function uploadDocument(
  file: File,
  knowledgeBaseId: string,
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

export async function listDocuments(params?: {
  knowledgeBaseId?: string;
  status?: DocumentStatus;
  limit?: number;
  offset?: number;
}): Promise<DocumentListResponse> {
  const qs = new URLSearchParams();
  if (params?.knowledgeBaseId) qs.set("knowledge_base_id", params.knowledgeBaseId);
  if (params?.status) qs.set("status", params.status);
  if (params?.limit != null) qs.set("limit", String(params.limit));
  if (params?.offset != null) qs.set("offset", String(params.offset));
  const query = qs.toString();

  const res = await fetch(`${BASE}/documents/${query ? `?${query}` : ""}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "list documents failed");
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

export async function retryGraph(
  documentId: string,
): Promise<GraphRetryResult> {
  const res = await fetch(`${BASE}/documents/${documentId}/retry-graph`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "retry graph failed");
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

// ── Sessions ────────────────────────────────────────

export async function listSessions(): Promise<Session[]> {
  const res = await fetch(`${BASE}/sessions/`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "list sessions failed");
  }
  return res.json();
}

export async function createSession(): Promise<Session> {
  const res = await fetch(`${BASE}/sessions/`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "create session failed");
  }
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "delete session failed");
  }
}

export interface SessionMessage {
  role: "user" | "assistant";
  content: string;
}

export async function getSessionMessages(
  sessionId: string,
): Promise<SessionMessage[]> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/messages`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "get session messages failed");
  }
  return res.json();
}

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
