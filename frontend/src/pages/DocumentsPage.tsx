import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import DocumentUpload from "@/components/DocumentUpload";
import DocumentList from "@/components/DocumentList";
import { getDocumentStatus, listDocuments } from "@/api/client";
import type { DocumentItem } from "@/types";

const KB_CONFIG: Record<string, { label: string; id: string }> = {
  book: { label: "书籍文献", id: "00000000-0000-0000-0000-000000000002" },
  regulation: { label: "法规", id: "00000000-0000-0000-0000-000000000003" },
};

function useKbDocs(kbId: string) {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDocs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listDocuments({ knowledgeBaseId: kbId, limit: 100 });
      setDocs(res.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => {
    void loadDocs();
  }, [loadDocs]);

  const onUploaded = useCallback(async (documentId: string) => {
    try {
      const d = await getDocumentStatus(documentId);
      setDocs((prev) => {
        const rest = prev.filter((x) => x.document_id !== d.document_id);
        return [d, ...rest];
      });
    } catch {
      // 刷新按钮可手动获取
    }
  }, []);

  return { docs, setDocs, loading, error, loadDocs, onUploaded };
}

export default function DocumentsPage() {
  const book = useKbDocs(KB_CONFIG.book.id);
  const regulation = useKbDocs(KB_CONFIG.regulation.id);

  const globalLoading = book.loading || regulation.loading;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 顶栏 */}
      <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">Documents</h2>
          <p className="text-xs text-gray-500 mt-0.5">按知识库分别上传与管理</p>
        </div>
        <button
          onClick={() => {
            void book.loadDocs();
            void regulation.loadDocs();
          }}
          disabled={globalLoading}
          className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-200 disabled:opacity-50 transition-colors"
          title="刷新所有列表"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${globalLoading ? "animate-spin" : ""}`} />
          刷新
        </button>
      </div>

      {/* 双列 KB 面板 */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* 书籍文献 */}
          <KBPanel
            label={KB_CONFIG.book.label}
            kbId={KB_CONFIG.book.id}
            docs={book.docs}
            loading={book.loading}
            error={book.error}
            onDocsChange={book.setDocs}
            onUploaded={book.onUploaded}
          />

          {/* 法规 */}
          <KBPanel
            label={KB_CONFIG.regulation.label}
            kbId={KB_CONFIG.regulation.id}
            docs={regulation.docs}
            loading={regulation.loading}
            error={regulation.error}
            onDocsChange={regulation.setDocs}
            onUploaded={regulation.onUploaded}
          />
        </div>
      </div>
    </div>
  );
}

function KBPanel({
  label,
  kbId,
  docs,
  loading,
  error,
  onDocsChange,
  onUploaded,
}: {
  label: string;
  kbId: string;
  docs: DocumentItem[];
  loading: boolean;
  error: string | null;
  onDocsChange: (fn: (prev: DocumentItem[]) => DocumentItem[]) => void;
  onUploaded: (id: string) => void;
}) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 flex flex-col min-h-0">
      {/* 面板标题 */}
      <h3 className="text-sm font-semibold text-gray-200 mb-4">{label}</h3>

      {/* 上传区 */}
      <DocumentUpload kbId={kbId} kbLabel={label} onUploaded={onUploaded} />

      {/* 分隔 */}
      <hr className="my-5 border-gray-800" />

      {/* 文档列表 */}
      <div className="flex-1 min-h-0">
        {error ? (
          <div className="flex items-center justify-center h-32">
            <p className="text-red-400/80 text-sm">{error}</p>
          </div>
        ) : loading && docs.length === 0 ? (
          <div className="flex items-center justify-center h-32">
            <p className="text-gray-500 text-sm">加载中…</p>
          </div>
        ) : (
          <DocumentList docs={docs} onDocsChange={onDocsChange} />
        )}
      </div>
    </div>
  );
}
