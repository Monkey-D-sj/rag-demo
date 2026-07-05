import { Download, GitBranch, RefreshCw, RotateCcw } from "lucide-react";
import { getDocumentStatus, retryDocument, retryGraph } from "@/api/client";
import { formatBytes, formatDate } from "@/lib/utils";
import type { DocumentItem } from "@/types";

const STATUS_MAP: Record<string, { label: string; cls: string }> = {
  pending:    { label: "待处理",  cls: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
  processing: { label: "处理中",  cls: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
  done:       { label: "已完成",  cls: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
  failed:     { label: "失败",    cls: "bg-red-500/10 text-red-400 border-red-500/20" },
};

const GRAPH_STATUS_MAP: Record<string, { label: string; cls: string }> = {
  pending:    { label: "待抽取",  cls: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
  processing: { label: "抽取中",  cls: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
  done:       { label: "已抽取",  cls: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
  failed:     { label: "抽取失败",cls: "bg-red-500/10 text-red-400 border-red-500/20" },
  skipped:    { label: "已跳过",  cls: "bg-gray-500/10 text-gray-400 border-gray-500/20" },
};

interface Props {
  docs: DocumentItem[];
  onDocsChange: (fn: (prev: DocumentItem[]) => DocumentItem[]) => void;
}

export default function DocumentList({ docs, onDocsChange }: Props) {
  const handleRefresh = async (id: string) => {
    try {
      const fresh = await getDocumentStatus(id);
      onDocsChange((prev) =>
        prev.map((x) => (x.document_id === id ? fresh : x)),
      );
    } catch {
      // ignore
    }
  };

  const handleRetry = async (id: string) => {
    try {
      await retryDocument(id);
      await handleRefresh(id);
    } catch {
      // ignore
    }
  };

  const handleRetryGraph = async (id: string) => {
    try {
      await retryGraph(id);
      await handleRefresh(id);
    } catch {
      // ignore
    }
  };

  if (docs.length === 0) {
    return (
      <div className="flex items-center justify-center h-48">
        <p className="text-gray-500 text-sm">暂无文档，上传后自动出现在这里</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 border-b border-gray-800">
            <th className="py-3 px-4 font-medium">文件名</th>
            <th className="py-3 px-4 font-medium">入库状态</th>
            <th className="py-3 px-4 font-medium">实体抽取</th>
            <th className="py-3 px-4 font-medium">大小</th>
            <th className="py-3 px-4 font-medium">Chunks</th>
            <th className="py-3 px-4 font-medium">创建时间</th>
            <th className="py-3 px-4 font-medium">操作</th>
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => {
            const st = STATUS_MAP[d.status] ?? STATUS_MAP.pending;
            const gs = GRAPH_STATUS_MAP[d.graph_status ?? ""];
            return (
              <tr key={d.document_id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="py-3 px-4 text-gray-200 max-w-[200px] truncate" title={d.filename}>
                  {d.filename}
                </td>
                <td className="py-3 px-4">
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${st.cls}`}>
                    {st.label}
                  </span>
                  {d.error && (
                    <span className="ml-2 text-xs text-red-400/70" title={d.error}>
                      {d.error.slice(0, 40)}{d.error.length > 40 ? "…" : ""}
                    </span>
                  )}
                </td>
                <td className="py-3 px-4">
                  {gs ? (
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${gs.cls}`}>
                      {gs.label}
                    </span>
                  ) : (
                    <span className="text-xs text-gray-600">—</span>
                  )}
                  {d.graph_error && (
                    <span className="ml-2 text-xs text-red-400/70" title={d.graph_error}>
                      {d.graph_error.slice(0, 40)}{d.graph_error.length > 40 ? "…" : ""}
                    </span>
                  )}
                </td>
                <td className="py-3 px-4 text-gray-400">
                  {d.size_bytes != null ? formatBytes(d.size_bytes) : "—"}
                </td>
                <td className="py-3 px-4 text-gray-400">
                  {d.chunk_count !== null ? d.chunk_count : "—"}
                </td>
                <td className="py-3 px-4 text-gray-400 whitespace-nowrap">
                  {formatDate(d.created_at)}
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleRefresh(d.document_id)}
                      className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors"
                      title="刷新"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                    </button>
                    {d.status === "done" && (
                      <a
                        href={`/api/documents/${d.document_id}/download`}
                        className="p-1.5 text-emerald-500 hover:text-emerald-400 transition-colors"
                        title="下载原文件"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </a>
                    )}
                    {d.status === "failed" && (
                      <button
                        onClick={() => handleRetry(d.document_id)}
                        className="p-1.5 text-amber-500 hover:text-amber-400 transition-colors"
                        title="强制重试入库"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {d.status === "done" && (d.graph_status === "failed" || d.graph_status === "skipped") && (
                      <button
                        onClick={() => handleRetryGraph(d.document_id)}
                        className="p-1.5 text-purple-500 hover:text-purple-400 transition-colors"
                        title="重试实体抽取"
                      >
                        <GitBranch className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
