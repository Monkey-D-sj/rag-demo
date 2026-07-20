// ── 与后端 /documents schema 对齐 ──────────────────────

export type DocumentStatus = "pending" | "processing" | "done" | "failed";

export interface DocumentItem {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  chunk_count: number | null;
  error: string | null;
  graph_status?: string | null;
  graph_error?: string | null;
  // 以下字段仅列表接口 GET /documents/ 返回;详情/上传接口不含,故可选
  knowledge_base_id?: string;
  content_type?: string;
  size_bytes?: number;
  created_at?: string;
  updated_at?: string;
}

export interface DocumentListResponse {
  total: number;
  limit: number;
  offset: number;
  items: DocumentItem[];
}

export interface RetryResult {
  document_id: string;
  status: string;
  message: string;
  retry_count: number | null;
}

export interface GraphRetryResult {
  document_id: string;
  graph_status: string;
  message: string;
}

// ── Session ──────────────────────────────────────────

export interface Session {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

// ── SSE 事件类型 ──────────────────────────────────────

/** 单条引用元数据 */
export interface Citation {
  index: number;
  /** 结构化引用标签，如 **《xxx法》第X条** */
  text: string;
  /** 原文片段，供弹窗预览 */
  snippet?: string;
  document_title: string;
}

/** 后端 /chat SSE 下发的事件的联合类型 */
export type ChatEvent =
  | { type: "status";    data: string }
  | { type: "message";   data: string }
  | { type: "error";     data: string }
  | { type: "citations"; data: Citation[] }
  | { type: "done";      data: null };

// ── LLM 调用统计 ────────────────────────────────────
export interface LlmSummaryRow {
  bucket: string;
  calls: number;
  success: number;
  failed: number;
  rejected: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
  avg_latency_ms: number;
  p95_latency_ms: number | null;
}

export interface LlmSummaryResponse {
  group_by: "day" | "model" | "source";
  rows: LlmSummaryRow[];
}
