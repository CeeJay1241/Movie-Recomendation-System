"use client";

import { useState } from "react";
import { MetricCard } from "@/components/metric-card";
import { StarRating } from "@/components/star-rating";
import { GenreBadges } from "@/components/genre-badge";
import TasteChart from "@/components/taste-chart";

interface HistoryRow {
  userId: number;
  movieId: number;
  title: string;
  genres: string;
  rating: number;
  rating_norm: number;
}

interface TasteRow {
  genre: string;
  avg_rating: number;
  count: number;
}

interface Personality {
  label: string;
  bias: number;
  fav_genre: string;
  fav_avg: number;
  most_watched: string;
  most_watched_count: number;
}

interface UserData {
  history: HistoryRow[];
  stats?: { movies_rated: number; avg_rating: number; highest_rated: string };
  taste?: TasteRow[];
  personality?: Personality;
}

export default function UserPage() {
  const [userId, setUserId] = useState(42);
  const [data, setData] = useState<UserData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function fetchUser() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`/data/users/${userId}.json`);
      if (!res.ok) throw new Error("Not found");
      const json = await res.json();
      setData(json);
    } catch {
      setError(`No data for user ${userId}. Try 1 to 610.`);
      setData(null);
    }
    setLoading(false);
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-bold tracking-tight mb-2">User profiles</h1>
        <p className="text-sm text-[var(--muted)]">
          Explore a user&apos;s rating history, taste profile, and rater personality.
        </p>
      </section>

      <div className="flex items-end gap-3">
        <div>
          <label htmlFor="uid" className="block text-xs text-[var(--muted)] mb-1">
            User ID
          </label>
          <input
            id="uid"
            type="number"
            min={1}
            max={610}
            value={userId}
            onChange={(e) => setUserId(Number(e.target.value))}
            onKeyDown={(e) => e.key === "Enter" && fetchUser()}
            className="bg-[var(--card)] border border-[var(--border)] rounded-md px-3 py-2 text-sm w-28 focus:outline-none focus:border-[var(--accent)] tabular-nums"
          />
        </div>
        <button
          onClick={fetchUser}
          disabled={loading}
          className="bg-[var(--accent)] hover:bg-[var(--accent-light)] text-white px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50"
        >
          {loading ? "Loading..." : "Load profile"}
        </button>
      </div>

      {error && <p className="text-sm text-[var(--error)]">{error}</p>}

      {data?.stats && (
        <>
          {/* Summary metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Movies rated" value={data.stats.movies_rated.toLocaleString()} />
            <MetricCard label="Avg rating" value={`${data.stats.avg_rating.toFixed(2)} ★`} />
            <MetricCard
              label="Highest rated"
              value={data.stats.highest_rated.length > 24
                ? data.stats.highest_rated.slice(0, 24) + "…"
                : data.stats.highest_rated}
            />
            {data.personality && (
              <MetricCard
                label="Rater style"
                value={data.personality.label}
                subtitle={`${data.personality.bias > 0 ? "+" : ""}${data.personality.bias.toFixed(2)}★ vs avg`}
              />
            )}
          </div>

          {/* Taste profile */}
          {data.taste && data.personality && (
            <section className="space-y-4">
              <h2 className="text-lg font-semibold">Taste profile</h2>
              <div className="grid md:grid-cols-3 gap-4 mb-4">
                <MetricCard label="Favorite genre" value={data.personality.fav_genre} subtitle={`${data.personality.fav_avg.toFixed(2)}★ avg`} />
                <MetricCard label="Most watched" value={data.personality.most_watched} subtitle={`${data.personality.most_watched_count} films`} />
                <MetricCard label="Genres explored" value={data.taste.length.toString()} />
              </div>
              <TasteChart taste={data.taste} />
            </section>
          )}

          {/* Rating history */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold">Rating history</h2>
            <div className="border border-[var(--border)] rounded-lg overflow-hidden max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0">
                  <tr className="bg-[var(--card)] text-[var(--muted)] text-xs uppercase tracking-wide">
                    <th className="text-left px-4 py-2">Title</th>
                    <th className="text-left px-4 py-2 hidden md:table-cell">Genres</th>
                    <th className="text-left px-4 py-2 w-32">Rating</th>
                    <th className="text-right px-4 py-2 w-16">Bias</th>
                  </tr>
                </thead>
                <tbody>
                  {data.history.map((row) => (
                    <tr key={row.movieId} className="border-t border-[var(--border)]">
                      <td className="px-4 py-2">{row.title}</td>
                      <td className="px-4 py-2 hidden md:table-cell">
                        <GenreBadges genres={row.genres} />
                      </td>
                      <td className="px-4 py-2">
                        <StarRating rating={row.rating} />
                      </td>
                      <td className={`px-4 py-2 text-right tabular-nums text-xs ${
                        row.rating_norm > 0
                          ? "text-[var(--success)]"
                          : row.rating_norm < 0
                          ? "text-[var(--error)]"
                          : "text-[var(--muted)]"
                      }`}>
                        {row.rating_norm > 0 ? "+" : ""}{row.rating_norm.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
