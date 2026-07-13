import type {
  DocumentItem,
  DocumentListResponse,
  DocumentStatus,
  GraphRetryResult,
  RetryResult,
} from "@/types";

const BASE = "/api";

/** 默认知识库 ID，与后端 rag.document.DEFAULT_KB_ID 保持一致 */
export const DEFAULT_KB_ID = "00000000-0000-0000-0000-000000000001";

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
