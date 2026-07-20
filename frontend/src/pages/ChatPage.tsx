import { useCallback, useEffect, useRef, useState } from "react";
import { MessageSquarePlus, Trash2 } from "lucide-react";
import ChatBox from "@/components/ChatBox";
import { listSessions, createSession, deleteSession } from "@/api/client";
import { formatDate, uuid } from "@/lib/utils";
import type { Session } from "@/types";

/** 本地兜底会话：当后端 session API 不可用时使用 */
function localSession(): Session {
  const id = uuid();
  return {
    session_id: id,
    title: "新会话",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fallbackRef = useRef(false);  // API 不可用时使用本地模式

  // 加载会话列表
  const load = useCallback(async () => {
    if (fallbackRef.current) return sessions;  // 本地模式下不需要重新加载
    try {
      const list = await listSessions();
      setSessions(list);
      return list;
    } catch {
      // API 不可用 → 进入本地兜底模式
      if (!fallbackRef.current) {
        fallbackRef.current = true;
        const fb = localSession();
        setSessions([fb]);
        return [fb];
      }
      return sessions;
    }
  }, [sessions]);

  // 初始化
  useEffect(() => {
    (async () => {
      const list = await load();
      if (list.length > 0 && !activeId) {
        setActiveId(list[0].session_id);
      }
      setLoading(false);
    })();
  // 仅在挂载时运行
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 创建新会话
  const handleCreate = async () => {
    if (fallbackRef.current) {
      const fb = localSession();
      setSessions((prev) => [fb, ...prev]);
      setActiveId(fb.session_id);
      return;
    }
    try {
      const s = await createSession();
      setSessions((prev) => [s, ...prev]);
      setActiveId(s.session_id);
    } catch {
      fallbackRef.current = true;
      const fb = localSession();
      setSessions((prev) => [fb, ...prev]);
      setActiveId(fb.session_id);
    }
  };

  // 删除会话
  const handleDelete = async (id: string) => {
    if (!fallbackRef.current) {
      try {
        await deleteSession(id);
      } catch {
        fallbackRef.current = true;
      }
    }
    const remaining = sessions.filter((s) => s.session_id !== id);
    setSessions(remaining);
    if (activeId === id) {
      if (remaining.length > 0) {
        setActiveId(remaining[0].session_id);
      } else {
        const fb = localSession();
        setSessions([fb]);
        setActiveId(fb.session_id);
      }
    }
  };

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
      {/* 侧边栏 */}
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
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-gray-200">Chat</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {activeId ? `Session: ${activeId.slice(0, 8)}…` : ""}
            </p>
          </div>
        </div>

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
