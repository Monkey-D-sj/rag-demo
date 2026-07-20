import { useCallback, useEffect, useState } from "react";
import { MessageSquarePlus, Trash2 } from "lucide-react";
import ChatBox from "@/components/ChatBox";
import { listSessions, createSession, deleteSession } from "@/api/client";
import { formatDate } from "@/lib/utils";
import type { Session } from "@/types";

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // 加载会话列表
  const load = useCallback(async () => {
    try {
      const list = await listSessions();
      setSessions(list);
      return list;
    } catch {
      return [];
    }
  }, []);

  // 初始化：加载列表 → 选中最近会话或创建新会话
  useEffect(() => {
    (async () => {
      const list = await load();
      if (list.length > 0) {
        setActiveId(list[0].session_id);
      } else {
        try {
          const s = await createSession();
          setSessions([s]);
          setActiveId(s.session_id);
        } catch {
          // 忽略
        }
      }
      setLoading(false);
    })();
  }, [load]);

  // 创建新会话
  const handleCreate = async () => {
    try {
      const s = await createSession();
      setSessions((prev) => [s, ...prev]);
      setActiveId(s.session_id);
    } catch {
      // 忽略
    }
  };

  // 删除会话
  const handleDelete = async (id: string) => {
    try {
      await deleteSession(id);
      const remaining = sessions.filter((s) => s.session_id !== id);
      setSessions(remaining);
      if (activeId === id) {
        if (remaining.length > 0) {
          setActiveId(remaining[0].session_id);
        } else {
          // 全删光了，自动创建新会话
          const s = await createSession();
          setSessions([s]);
          setActiveId(s.session_id);
        }
      }
    } catch {
      // 忽略
    }
  };

  // ChatBox 发首条消息后刷新标题
  const handleFirstMessage = useCallback(() => {
    load().then(setSessions);
  }, [load]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-gray-500 text-sm">加载中…</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex min-h-0">
      {/* 侧边栏：会话列表 */}
      <aside className="w-60 border-r border-gray-800 flex flex-col min-h-0 shrink-0">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            会话
          </span>
          <button
            onClick={handleCreate}
            className="p-1.5 text-gray-500 hover:text-emerald-400 hover:bg-emerald-500/10 rounded-lg transition-colors"
            title="新建会话"
          >
            <MessageSquarePlus className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-1">
          {sessions.map((s) => (
            <div
              key={s.session_id}
              onClick={() => setActiveId(s.session_id)}
              className={`group flex items-center px-4 py-2.5 cursor-pointer transition-colors ${
                s.session_id === activeId
                  ? "bg-emerald-500/10 border-r-2 border-emerald-500"
                  : "hover:bg-gray-800/50 border-r-2 border-transparent"
              }`}
            >
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm truncate ${
                    s.session_id === activeId ? "text-emerald-400" : "text-gray-300"
                  }`}
                >
                  {s.title}
                </p>
                <p className="text-xs text-gray-600 mt-0.5">
                  {formatDate(s.updated_at)}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(s.session_id);
                }}
                className="p-1 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all rounded"
                title="删除会话"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* 主区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 顶栏 */}
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-gray-200">Chat</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {activeId ? `Session: ${activeId.slice(0, 8)}…` : ""}
            </p>
          </div>
        </div>

        {/* ChatBox */}
        {activeId && (
          <ChatBox
            key={activeId}
            sessionId={activeId}
            className="flex-1"
            onFirstMessage={handleFirstMessage}
          />
        )}
      </div>
    </div>
  );
}
