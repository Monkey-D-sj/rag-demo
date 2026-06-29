import { useEffect, useState } from "react";
import { getHealth } from "@/api/client";
import { CheckCircle2, XCircle } from "lucide-react";

export default function HealthPage() {
  const [health, setHealth] = useState<{
    status: string;
    checks: Record<string, string>;
  } | null>(null);
  const [error, setError] = useState("");

  const refresh = async () => {
    setError("");
    try {
      const h = await getHealth();
      setHealth(h);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "获取健康状态失败");
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">Health</h2>
          <p className="text-xs text-gray-500 mt-0.5">服务连通性检查</p>
        </div>
        <button
          onClick={refresh}
          className="px-3 py-1.5 text-xs bg-gray-800 border border-gray-700 rounded-lg
                     text-gray-300 hover:bg-gray-700 transition-colors"
        >
          刷新
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {error && (
          <p className="text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-lg px-4 py-3 mb-4">
            {error}
          </p>
        )}

        {health && (
          <div className="space-y-3">
            {Object.entries(health.checks).map(([name, status]) => {
              const ok = status === "ok";
              return (
                <div
                  key={name}
                  className="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-3"
                >
                  <span className="text-sm text-gray-300 capitalize">{name}</span>
                  <div className="flex items-center gap-2">
                    {ok ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-400" />
                    )}
                    <span
                      className={`text-xs font-mono ${ok ? "text-emerald-400" : "text-red-400"}`}
                    >
                      {status}
                    </span>
                  </div>
                </div>
              );
            })}

            <div className="pt-2 text-center">
              <span
                className={`text-xs px-3 py-1 rounded-full border ${
                  health.status === "ok"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                    : "bg-red-500/10 text-red-400 border-red-500/20"
                }`}
              >
                {health.status === "ok" ? "全部正常" : "部分异常"}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
