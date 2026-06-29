import { useCallback, useState } from "react";
import DocumentUpload from "@/components/DocumentUpload";
import DocumentList from "@/components/DocumentList";
import { getDocumentStatus } from "@/api/client";
import type { DocumentItem } from "@/types";

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);

  const onUploaded = useCallback(async (documentId: string) => {
    try {
      const d = await getDocumentStatus(documentId);
      setDocs((prev) => [...prev, d]);
    } catch {
      // 列表轮询会自动补上
    }
  }, []);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 顶栏 */}
      <div className="px-6 py-4 border-b border-gray-800">
        <h2 className="text-sm font-semibold text-gray-200">Documents</h2>
        <p className="text-xs text-gray-500 mt-0.5">上传知识库文档，查看处理状态</p>
      </div>

      {/* 内容 */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <DocumentUpload onUploaded={onUploaded} />
        <DocumentList docs={docs} onDocsChange={setDocs} />
      </div>
    </div>
  );
}
