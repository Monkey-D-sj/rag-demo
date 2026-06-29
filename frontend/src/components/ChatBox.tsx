import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamChat } from "@/api/client";

interface Message {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
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

  // 自动滚底
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");
    setSending(true);

    // 占位：assistant 逐 token 填充
    const assistantIdx = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", isStreaming: true },
    ]);

    try {
      for await (const ev of streamChat(sessionId, q)) {
        setMessages((prev) => {
          const next = [...prev];
          const msg = next[assistantIdx];
          if (!msg) return prev;

          switch (ev.type) {
            case "generate":
              next[assistantIdx] = { ...msg, content: msg.content + ev.data };
              break;
            case "error":
              next[assistantIdx] = {
                ...msg,
                content: msg.content + `\n\n> ⚠️ ${ev.data}`,
                isStreaming: false,
              };
              break;
            case "done":
              next[assistantIdx] = { ...msg, isStreaming: false };
              break;
          }
          return next;
        });

        if (ev.type === "done" || ev.type === "error") break;
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "unknown error";
      setMessages((prev) => {
        const next = [...prev];
        next[assistantIdx] = {
          role: "assistant",
          content: `> ❌ 请求失败: ${msg}`,
          isStreaming: false,
        };
        return next;
      });
    } finally {
      setSending(false);
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
                  <ReactMarkdown>{m.content}</ReactMarkdown>
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
