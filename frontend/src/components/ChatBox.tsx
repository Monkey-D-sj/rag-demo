import { Component, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { createChatStream } from "@/api/stream";

// 将文本中 [n] 引用标记拆分为 React 节点（用于流式 / 降级渲染）
function renderContent(text: string): React.ReactNode[] {
  const parts = text.split(/(\[\d+\])/);
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    return m ? <sup key={i}>[{m[1]}]</sup> : part;
  });
}

// 转义 [n] 防止 ReactMarkdown 把它当成链接引用
function escapeCitations(text: string): string {
  return text.replace(/\[(\d+)\]/g, "\\[$1\\]");
}

// ReactMarkdown 解析失败时降级为纯文本（含上标）
class MarkdownSafe extends Component<{ content: string }> {
  state = { error: false };

  static getDerivedStateFromError() {
    return { error: true };
  }

  render() {
    if (this.state.error) {
      return <span className="whitespace-pre-wrap">{renderContent(this.props.content)}</span>;
    }
    return <ReactMarkdown>{escapeCitations(this.props.content)}</ReactMarkdown>;
  }
}

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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");
    setSending(true);

    const assistantIdx = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", isStreaming: true },
    ]);

    let eventCount = 0;
    try {
      for await (const ev of createChatStream(sessionId, q)) {
        eventCount++;
        if (eventCount <= 5 || ev.type === "done" || ev.type === "error") {
          console.log(`[SSE #${eventCount}]`, ev.type, ev.type === "message" ? `"${ev.data}"` : ev.data);
        }
        setMessages((prev) => {
          const next = [...prev];
          const msg = next[assistantIdx];
          if (!msg) return prev;

          switch (ev.type) {
            case "status":
              next[assistantIdx] = { ...msg, status: ev.data };
              break;
            case "message":
              next[assistantIdx] = {
                ...msg,
                content: msg.content + ev.data,
              };
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
      console.log(`[SSE] stream complete, total events: ${eventCount}`);
    } catch (err: unknown) {
      console.error("[SSE] stream error:", err);
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
                    <span className="whitespace-pre-wrap">{renderContent(m.content)}</span>
                  ) : (
                    <MarkdownSafe content={m.content} />
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
