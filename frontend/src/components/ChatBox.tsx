import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { createChatStream } from "@/api/stream";

interface Message {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  status?: string;
}

export default function ChatBox({
  sessionId,
  className,
}: {
  sessionId: string;
  className?: string;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // 用 ref 累积流式内容，避免 100+ 次 setState
  const streamContentRef = useRef("");
  const streamStatusRef = useRef("");
  const streamMsgIdxRef = useRef(-1);
  const rafRef = useRef(0);

  const flushStream = useCallback(() => {
    const idx = streamMsgIdxRef.current;
    if (idx < 0) return;
    setMessages((prev) => {
      const next = [...prev];
      if (!next[idx]) return prev;
      next[idx] = {
        ...next[idx],
        content: streamContentRef.current,
        status: streamStatusRef.current || next[idx].status,
      };
      return next;
    });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");
    setSending(true);

    // 占位
    const assistantIdx = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", isStreaming: true },
    ]);

    // 初始化 ref
    streamContentRef.current = "";
    streamStatusRef.current = "";
    streamMsgIdxRef.current = assistantIdx;

    // 定时刷新：每 60ms 把累积内容同步到 state
    const tick = () => {
      flushStream();
      rafRef.current = requestAnimationFrame(() => {
        tick();
      });
    };
    rafRef.current = requestAnimationFrame(() => {
      tick();
    });

    try {
      for await (const ev of createChatStream(sessionId, q)) {
        switch (ev.type) {
          case "status":
            streamStatusRef.current = ev.data;
            break;
          case "message":
            streamContentRef.current += ev.data;
            break;
          case "error":
            streamContentRef.current += `\n\n> ⚠️ ${ev.data}`;
            break;
          case "done":
            break;
        }
        if (ev.type === "done" || ev.type === "error") break;
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "unknown error";
      streamContentRef.current = `> ❌ 请求失败: ${msg}`;
    } finally {
      cancelAnimationFrame(rafRef.current);
      setSending(false);
      // 最后一次同步：更新 content 并取消 streaming 标记
      const idx = streamMsgIdxRef.current;
      const finalContent = streamContentRef.current;
      const finalStatus = streamStatusRef.current;
      setMessages((prev) => {
        const next = [...prev];
        if (next[idx]) {
          next[idx] = {
            ...next[idx],
            content: finalContent,
            status: finalStatus || next[idx].status,
            isStreaming: false,
          };
        }
        return next;
      });
    }
  };

  return (
    <div className={`flex flex-col h-full ${className ?? ""}`}>
      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500 text-sm">输入问题开始对话</p>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-emerald-500/15 border border-emerald-500/20 text-gray-100"
                  : "bg-gray-800/80 text-gray-200 prose-chat"
              }`}
            >
              {m.role === "assistant" ? (
                <>
                  {m.status && (
                    <p className="text-xs text-gray-500 mb-1">{m.status}</p>
                  )}
                  {m.isStreaming ? (
                    <span className="whitespace-pre-wrap">{m.content}</span>
                  ) : (
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  )}
                  {m.isStreaming && (
                    <span className="inline-block w-2 h-4 bg-emerald-400 ml-0.5 animate-pulse align-text-bottom" />
                  )}
                </>
              ) : (
                <p>{m.content}</p>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* 输入框 */}
      <div className="border-t border-gray-800 p-4">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={sending}
            placeholder="输入你的问题… (Enter 发送)"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm
                       placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50
                       disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={sending || !input.trim()}
            className="px-5 py-3 bg-emerald-500 text-gray-950 font-semibold text-sm rounded-xl
                       hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-colors"
          >
            {sending ? "…" : "发送"}
          </button>
        </div>
      </div>
    </div>
  );
}
