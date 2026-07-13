import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import DocumentUpload from "@/components/DocumentUpload";
import DocumentList from "@/components/DocumentList";
import { DEFAULT_KB_ID, getDocumentStatus, listDocuments } from "@/api/client";
import type { DocumentItem } from "@/types";

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDocs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listDocuments({ knowledgeBaseId: DEFAULT_KB_ID, limit: 100 });
      setDocs(res.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载文档列表失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDocs();
  }, [loadDocs]);

  const onUploaded = useCallback(async (documentId: string) => {
    try {
      const d = await getDocumentStatus(documentId);
      // 去重:已存在则替换,否则插到最前(与列表的创建时间倒序一致)
      setDocs((prev) => {
        const rest = prev.filter((x) => x.document_id !== d.document_id);
        return [d, ...rest];
      });
    } catch {
      // 刷新按钮可手动获取
    }
  }, []);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 顶栏 */}
      <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">Documents</h2>
          <p className="text-xs text-gray-500 mt-0.5">上传知识库文档，查看处理状态</p>
        </div>
        <button
          onClick={() => void loadDocs()}
          disabled={loading}
          className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-200 disabled:opacity-50 transition-colors"
          title="刷新列表"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          刷新
        </button>
      </div>

      {/* 内容 */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <DocumentUpload onUploaded={onUploaded} knowledgeBaseId={DEFAULT_KB_ID} />
        {error ? (
          <div className="flex items-center justify-center h-48">
            <p className="text-red-400/80 text-sm">{error}</p>
          </div>
        ) : loading && docs.length === 0 ? (
          <div className="flex items-center justify-center h-48">
            <p className="text-gray-500 text-sm">加载中…</p>
          </div>
        ) : (
          <DocumentList docs={docs} onDocsChange={setDocs} />
        )}
      </div>
    </div>
  );
}
