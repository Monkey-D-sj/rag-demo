import { Component, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import { createChatStream } from "@/api/stream";
import type { Citation } from "@/types";

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

// 将 [n] 替换为 HTML <cite> 标签，供 rehype-raw 解析后由 components 映射
function injectCitationHtml(text: string): string {
  return text.replace(/\[(\d+)\]/g, '<cite data-idx="$1"></cite>');
}

// 点击外部关闭 hook
function useClickOutside(ref: React.RefObject<HTMLElement | null>, handler: () => void) {
  useEffect(() => {
    const listener = (e: MouseEvent) => {
      if (!ref.current || ref.current.contains(e.target as Node)) return;
      handler();
    };
    document.addEventListener("mousedown", listener);
    return () => document.removeEventListener("mousedown", listener);
  }, [ref, handler]);
}

// ReactMarkdown 解析失败时降级为纯文本（含上标）
class MarkdownSafe extends Component<{ content: string; citations?: Citation[] }> {
  state = { error: false };

  static getDerivedStateFromError() {
    return { error: true };
  }

  render() {
    const { content, citations } = this.props;
    if (this.state.error) {
      return <span className="whitespace-pre-wrap">{renderContent(content)}</span>;
    }
    // 有引用数据时：将 [n] 转换为 HTML cite 标签 → rehype-raw 解析 → components 映射
    if (citations && citations.length > 0) {
      return (
        <ReactMarkdown
          rehypePlugins={[rehypeRaw]}
          components={{
            cite: ({ 'data-idx': idx }: any) => (
              <CitationBadgeInline idx={Number(idx)} citations={citations} />
            ),
          }}
        >
          {injectCitationHtml(content)}
        </ReactMarkdown>
      );
    }
    return <ReactMarkdown>{escapeCitations(content)}</ReactMarkdown>;
  }
}

// 内联引用徽章（在 MarkdownSafe 的 components 中使用，不经过 text split）
function CitationBadgeInline({ idx, citations }: { idx: number; citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useClickOutside(ref, () => setOpen(false));

  const cit = citations.find((c) => c.index === idx);
  if (!cit) return <sup>[{idx}]</sup>;

  // 弹窗优先展示原文片段 snippet，回退到 text（结构化标签）
  const raw = cit.snippet || cit.text;
  const preview = raw.length > 300 ? raw.slice(0, 300) + "…" : raw;

  return (
    <span ref={ref} className="relative inline-block">
      <sup
        onClick={() => setOpen((v) => !v)}
        className="cursor-pointer text-emerald-400 hover:text-emerald-300 hover:underline transition-colors"
        title={`来源: ${cit.document_title}`}
      >
        [{idx}]
      </sup>
      {open && (
        <div className="absolute bottom-full left-0 mb-2 w-80 max-h-64 overflow-y-auto rounded-xl border border-gray-700 bg-gray-900 shadow-2xl z-50 p-3 text-xs leading-relaxed">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-emerald-400 font-semibold">[{idx}]</span>
            <span className="text-gray-400">{cit.document_title}</span>
          </div>
          <p className="text-gray-300 whitespace-pre-wrap">{preview}</p>
        </div>
      )}
    </span>
  );
}

interface Message {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  status?: string;
  citations?: Citation[];
}

export default function ChatBox({
  sessionId,
  className,
  onFirstMessage,
}: {
  sessionId: string;
  className?: string;
  /** 首个用户消息发送后回调，用于刷新会话列表标题 */
  onFirstMessage?: () => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const firstMessageSent = useRef(false);
  const citationsRef = useRef<Citation[]>([]);
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

    // 首条消息通知父组件刷新标题
    if (!firstMessageSent.current) {
      firstMessageSent.current = true;
      onFirstMessage?.();
    }

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
            case "citations":
              citationsRef.current = ev.data;
              next[assistantIdx] = { ...msg, citations: ev.data };
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
    <div className={`flex flex-col h-full min-h-0 ${className ?? ""}`}>
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
                    <MarkdownSafe
                      content={m.content}
                      citations={m.citations}
                    />
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
