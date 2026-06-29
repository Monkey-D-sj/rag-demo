import { type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { MessageSquare, FileText, HeartPulse } from "lucide-react";

const NAV = [
  { to: "/", label: "Chat", icon: MessageSquare },
  { to: "/documents", label: "Documents", icon: FileText },
  { to: "/health", label: "Health", icon: HeartPulse },
] as const;

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen">
      {/* 侧栏 */}
      <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="px-5 py-4 border-b border-gray-800">
          <h1 className="text-lg font-bold tracking-tight">
            <span className="text-emerald-400">RAG</span> Demo
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">知识库问答系统</p>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                [
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                  isActive
                    ? "bg-emerald-500/10 text-emerald-400"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200",
                ].join(" ")
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* 主区域 */}
      <main className="flex-1 flex flex-col overflow-hidden">{children}</main>
    </div>
  );
}
