"use client";

import { useEffect, useState } from "react";
import { MetricCard } from "@/components/metric-card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Metrics {
  rmse: number;
  mae: number;
  precision_at_k: number;
  recall_at_k: number;
  k: number;
  threshold: number;
}

export default function PerformancePage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => {
    fetch("/data/metrics.json")
      .then((r) => r.json())
      .then(setMetrics);
  }, []);

  if (!metrics) {
    return (
      <p className="text-sm text-[var(--muted)]">Loading metrics...</p>
    );
  }

  const chartData = [
    { name: "RMSE", value: metrics.rmse, type: "error" },
    { name: "MAE", value: metrics.mae, type: "error" },
    { name: `Precision@${metrics.k}`, value: metrics.precision_at_k, type: "ranking" },
    { name: `Recall@${metrics.k}`, value: metrics.recall_at_k, type: "ranking" },
  ];

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-bold tracking-tight mb-2">
          Model performance
        </h1>
        <p className="text-sm text-[var(--muted)]">
          Evaluated on a held-out 20% test split (random seed 42). Liked
          threshold: ≥ {metrics.threshold} ★.
        </p>
      </section>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="RMSE"
          value={metrics.rmse.toFixed(4)}
          subtitle="Root mean squared error"
        />
        <MetricCard
          label="MAE"
          value={metrics.mae.toFixed(4)}
          subtitle="Mean absolute error"
        />
        <MetricCard
          label={`Precision@${metrics.k}`}
          value={`${(metrics.precision_at_k * 100).toFixed(1)}%`}
          subtitle="Fraction of top-10 liked"
        />
        <MetricCard
          label={`Recall@${metrics.k}`}
          value={`${(metrics.recall_at_k * 100).toFixed(1)}%`}
          subtitle="Liked items in top-10"
        />
      </div>

      {/* Chart */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-6">
        <p className="text-xs text-[var(--muted)] uppercase tracking-wide mb-4">
          Evaluation metrics
        </p>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
            <XAxis
              dataKey="name"
              tick={{ fill: "#fafafa", fontSize: 12 }}
              axisLine={{ stroke: "#27272a" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              axisLine={{ stroke: "#27272a" }}
              tickLine={false}
              domain={[0, (dataMax: number) => Math.ceil(dataMax * 1.3 * 10) / 10]}
            />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #27272a",
                borderRadius: 6,
                fontSize: 12,
              }}
              formatter={(value) => [Number(value).toFixed(4), "Score"]}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={48}>
              {chartData.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={entry.type === "error" ? "#ef4444" : "#22c55e"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-[var(--muted)] text-center mt-2">
          Red = error metrics (lower is better) · Green = ranking metrics (higher
          is better)
        </p>
      </div>

      {/* Explanations */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-5 space-y-3 text-sm text-[var(--muted)]">
          <p className="text-[var(--foreground)] font-medium">Error metrics</p>
          <p>
            <strong className="text-[var(--foreground)]">RMSE</strong> — On
            average, predicted ratings are off by {metrics.rmse.toFixed(2)} stars.
            Sensitive to large individual errors.
          </p>
          <p>
            <strong className="text-[var(--foreground)]">MAE</strong> — The
            typical prediction is within {metrics.mae.toFixed(2)} stars of the
            actual rating. Less sensitive to outliers than RMSE.
          </p>
        </div>
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-5 space-y-3 text-sm text-[var(--muted)]">
          <p className="text-[var(--foreground)] font-medium">Ranking metrics</p>
          <p>
            <strong className="text-[var(--foreground)]">
              Precision@{metrics.k}
            </strong>{" "}
            — Of the {metrics.k} movies recommended,{" "}
            {(metrics.precision_at_k * 100).toFixed(1)}% were ones the user
            actually liked (rated ≥ {metrics.threshold}★).
          </p>
          <p>
            <strong className="text-[var(--foreground)]">
              Recall@{metrics.k}
            </strong>{" "}
            — Of all movies a user liked,{" "}
            {(metrics.recall_at_k * 100).toFixed(1)}% appeared in their
            top-{metrics.k} recommendations.
          </p>
        </div>
      </div>
    </div>
  );
}
