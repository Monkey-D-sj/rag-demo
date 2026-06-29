// ── 与后端 /documents schema 对齐 ──────────────────────

export type DocumentStatus = "pending" | "processing" | "done" | "failed";

export interface DocumentItem {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  chunk_count: number | null;
  error: string | null;
}

export interface RetryResult {
  document_id: string;
  status: string;
  message: string;
  retry_count: number | null;
}

// ── SSE 事件类型 ──────────────────────────────────────

/** 后端 /chat SSE 下发的事件的联合类型 */
export type ChatEvent =
  | { type: "query"; data: string }
  | { type: "recall"; data: unknown }
  | { type: "generate"; data: string }
  | { type: "error"; data: string }
  | { type: "done"; data: null };
