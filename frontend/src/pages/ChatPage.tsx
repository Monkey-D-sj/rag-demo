import { useMemo } from "react";
import ChatBox from "@/components/ChatBox";
import { uuid } from "@/lib/utils";

export default function ChatPage() {
  // 会话 ID 页面级别持有（刷新页面 = 新会话）
  const sessionId = useMemo(() => uuid(), []);

  return (
    <div className="flex-1 flex flex-col">
      {/* 顶栏 */}
      <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">Chat</h2>
          <p className="text-xs text-gray-500 mt-0.5">Session: {sessionId.slice(0, 8)}…</p>
        </div>
      </div>

      {/* ChatBox 撑满剩余高度 */}
      <ChatBox sessionId={sessionId} className="flex-1" />
    </div>
  );
}
