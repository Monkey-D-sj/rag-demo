import { useEffect, useState } from "react";
import { getLlmSummary } from "@/api/client";
import type { LlmSummaryRow } from "@/types";

function fmtCost(v: number): string {
  return `¥${Number(v).toFixed(4)}`;
}

function fmtDay(bucket: string): string {
  // bucket 形如 "2026-07-17T00:00:00+00:00" → "07-17"
  return bucket.slice(5, 10);
}

/** 汇总卡片 */
function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xl font-semibold text-gray-100 mt-1">{value}</div>
    </div>
  );
}

/** 按天调用量横条(单色系,拒绝量叠加为警示色) */
function DayTrend({ rows }: { rows: LlmSummaryRow[] }) {
  const max = Math.max(1, ...rows.map((r) => r.calls));
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.bucket} className="flex items-center gap-2 text-xs">
          <span className="w-12 text-gray-500 shrink-0">{fmtDay(r.bucket)}</span>
          <div className="flex-1 h-4 bg-gray-800 rounded overflow-hidden flex">
            <div
              className="h-full bg-emerald-500/70"
              style={{ width: `${((r.calls - r.rejected) / max) * 100}%` }}
            />
            <div
              className="h-full bg-amber-500/70"
              style={{ width: `${(r.rejected / max) * 100}%` }}
            />
          </div>
          <span className="w-24 text-right text-gray-400 shrink-0">
            {r.calls} 次 / {fmtCost(r.cost)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function StatsPage() {
  const [byDay, setByDay] = useState<LlmSummaryRow[]>([]);
  const [byModel, setByModel] = useState<LlmSummaryRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getLlmSummary("day"), getLlmSummary("model")])
      .then(([day, model]) => {
        setByDay(day.rows);
        setByModel(model.rows);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const totals = byDay.reduce(
    (acc, r) => ({
      calls: acc.calls + r.calls,
      success: acc.success + r.success,
      rejected: acc.rejected + r.rejected,
      cost: acc.cost + Number(r.cost),
      tokens: acc.tokens + r.input_tokens + r.output_tokens,
    }),
    { calls: 0, success: 0, rejected: 0, cost: 0, tokens: 0 },
  );
  const successRate =
    totals.calls > 0 ? `${((totals.success / totals.calls) * 100).toFixed(1)}%` : "-";

  return (
    <div className="p-6 overflow-y-auto space-y-6">
      <h2 className="text-lg font-semibold">LLM 调用统计(近 14 天)</h2>
      {error && <p className="text-sm text-red-400">加载失败: {error}</p>}

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard label="总调用" value={String(totals.calls)} />
        <StatCard label="成功率" value={successRate} />
        <StatCard label="被拒绝(限流/熔断)" value={String(totals.rejected)} />
        <StatCard label="Token 合计" value={totals.tokens.toLocaleString()} />
        <StatCard label="成本合计" value={fmtCost(totals.cost)} />
      </div>

      <section className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">按天调用量与成本</h3>
        {byDay.length === 0 ? (
          <p className="text-sm text-gray-500">暂无数据</p>
        ) : (
          <DayTrend rows={byDay} />
        )}
      </section>

      <section className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">按模型分布</h3>
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-gray-500 border-b border-gray-800">
            <tr>
              <th className="py-2">模型</th>
              <th className="py-2 text-right">调用</th>
              <th className="py-2 text-right">成功/失败/拒绝</th>
              <th className="py-2 text-right">Tokens(入/出)</th>
              <th className="py-2 text-right">P95 延迟</th>
              <th className="py-2 text-right">成本</th>
            </tr>
          </thead>
          <tbody>
            {byModel.map((r) => (
              <tr key={r.bucket} className="border-b border-gray-800/50 text-gray-300">
                <td className="py-2">{r.bucket}</td>
                <td className="py-2 text-right">{r.calls}</td>
                <td className="py-2 text-right">
                  {r.success}/{r.failed}/{r.rejected}
                </td>
                <td className="py-2 text-right">
                  {r.input_tokens.toLocaleString()}/{r.output_tokens.toLocaleString()}
                </td>
                <td className="py-2 text-right">
                  {r.p95_latency_ms != null ? `${Math.round(r.p95_latency_ms)}ms` : "-"}
                </td>
                <td className="py-2 text-right">{fmtCost(r.cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
