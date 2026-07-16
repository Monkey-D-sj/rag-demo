import { useCallback, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import { uploadDocument } from "@/api/client";
import { formatBytes } from "@/lib/utils";

const MAX_MB = 20;
const ALLOWED = ["txt", "md", "pdf", "docx"];

interface Props {
  kbId: string;
  kbLabel: string;
  onUploaded: (id: string) => void;
}

export default function DocumentUpload({ kbId, kbLabel, onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = useCallback((f: File): string | null => {
    const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
    if (!ALLOWED.includes(ext)) return `不支持的文件类型: .${ext}`;
    if (f.size > MAX_MB * 1024 * 1024) return `文件超过 ${MAX_MB}MB 上限`;
    return null;
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (!f) return;
      const msg = validate(f);
      if (msg) {
        setError(msg);
        return;
      }
      setError("");
      setFile(f);
    },
    [validate],
  );

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const { document_id } = await uploadDocument(file, kbId);
      setFile(null);
      onUploaded(document_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-3">
      {/* KB 标签（只读） */}
      <p className="text-xs text-gray-500">
        上传至 <span className="text-gray-300 font-medium">{kbLabel}</span>
      </p>

      {/* 拖拽区 */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-gray-700 rounded-xl p-6 text-center
                   cursor-pointer hover:border-emerald-500/50 hover:bg-gray-800/50
                   transition-colors"
      >
        <Upload className="w-6 h-6 mx-auto mb-2 text-gray-500" />
        <p className="text-sm text-gray-400">
          拖拽文件或 <span className="text-emerald-400">点击选择</span>
        </p>
        <p className="text-xs text-gray-600 mt-1">
          txt / md / pdf / docx（最大 {MAX_MB}MB）
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".txt,.md,.pdf,.docx"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const msg = validate(f);
            if (msg) setError(msg);
            else {
              setError("");
              setFile(f);
            }
          }}
        />
      </div>

      {/* 已选文件 */}
      {file && (
        <div className="flex items-center justify-between bg-gray-800 rounded-lg px-4 py-3">
          <div className="min-w-0">
            <p className="text-sm text-gray-200 truncate">{file.name}</p>
            <p className="text-xs text-gray-500">{formatBytes(file.size)}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="px-4 py-1.5 bg-emerald-500 text-gray-950 text-sm font-medium rounded-lg
                         hover:bg-emerald-400 disabled:opacity-40 transition-colors"
            >
              {uploading ? "上传中…" : "上传"}
            </button>
            <button
              onClick={() => setFile(null)}
              className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {error && (
        <p className="text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}
    </div>
  );
}
