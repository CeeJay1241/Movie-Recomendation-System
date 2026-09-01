"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface TasteRow {
  genre: string;
  avg_rating: number;
  count: number;
}

function ratingColor(rating: number): string {
  if (rating >= 4.0) return "#22c55e";
  if (rating >= 3.5) return "#6366f1";
  if (rating >= 3.0) return "#f59e0b";
  return "#ef4444";
}

export default function TasteChart({ taste }: { taste: TasteRow[] }) {
  const sorted = [...taste].sort((a, b) => a.avg_rating - b.avg_rating);

  return (
    <div className="grid md:grid-cols-2 gap-6">
      {/* Avg rating by genre */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
        <p className="text-xs text-[var(--muted)] uppercase tracking-wide mb-3">
          Avg rating by genre
        </p>
        <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 28)}>
          <BarChart data={sorted} layout="vertical" margin={{ left: 60, right: 40 }}>
            <XAxis type="number" domain={[0, 5]} tick={{ fill: "#a1a1aa", fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="genre"
              tick={{ fill: "#fafafa", fontSize: 11 }}
              width={55}
            />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #27272a",
                borderRadius: 6,
                fontSize: 12,
              }}
              formatter={(value) => [`${Number(value).toFixed(2)} ★`, "Avg rating"]}
            />
            <Bar dataKey="avg_rating" radius={[0, 4, 4, 0]}>
              {sorted.map((entry) => (
                <Cell key={entry.genre} fill={ratingColor(entry.avg_rating)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Watch count by genre */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
        <p className="text-xs text-[var(--muted)] uppercase tracking-wide mb-3">
          Movies watched per genre
        </p>
        <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 28)}>
          <BarChart
            data={[...taste].sort((a, b) => a.count - b.count)}
            layout="vertical"
            margin={{ left: 60, right: 40 }}
          >
            <XAxis type="number" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="genre"
              tick={{ fill: "#fafafa", fontSize: 11 }}
              width={55}
            />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #27272a",
                borderRadius: 6,
                fontSize: 12,
              }}
              formatter={(value) => [value, "Movies"]}
            />
            <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
