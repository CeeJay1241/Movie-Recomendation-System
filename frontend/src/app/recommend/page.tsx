"use client";

import { useState } from "react";
import { RatingBar } from "@/components/star-rating";
import { GenreBadges } from "@/components/genre-badge";

interface Rec {
  movieId: number;
  title: string;
  genres: string;
  est_rating: number;
}

export default function RecommendPage() {
  const [userId, setUserId] = useState(42);
  const [recs, setRecs] = useState<Rec[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function fetchRecs() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`/data/users/${userId}.json`);
      if (!res.ok) throw new Error("User not found");
      const data = await res.json();
      setRecs(data.recommendations || []);
    } catch {
      setError(`No data found for user ${userId}. Try a number between 1 and 610.`);
      setRecs(null);
    }
    setLoading(false);
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-bold tracking-tight mb-2">
          Get recommendations
        </h1>
        <p className="text-sm text-[var(--muted)]">
          Enter a user ID (1 to 610) to see their top-10 predicted films based on
          SVD collaborative filtering.
        </p>
      </section>

      <div className="flex items-end gap-3">
        <div>
          <label
            htmlFor="userId"
            className="block text-xs text-[var(--muted)] mb-1"
          >
            User ID
          </label>
          <input
            id="userId"
            type="number"
            min={1}
            max={610}
            value={userId}
            onChange={(e) => setUserId(Number(e.target.value))}
            onKeyDown={(e) => e.key === "Enter" && fetchRecs()}
            className="bg-[var(--card)] border border-[var(--border)] rounded-md px-3 py-2 text-sm w-28 focus:outline-none focus:border-[var(--accent)] tabular-nums"
          />
        </div>
        <button
          onClick={fetchRecs}
          disabled={loading}
          className="bg-[var(--accent)] hover:bg-[var(--accent-light)] text-white px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50"
        >
          {loading ? "Loading..." : "Recommend"}
        </button>
      </div>

      {error && (
        <p className="text-sm text-[var(--error)]">{error}</p>
      )}

      {recs && recs.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm text-[var(--muted)]">
            Top {recs.length} for User {userId}
          </h2>
          <div className="border border-[var(--border)] rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[var(--card)] text-[var(--muted)] text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-3 w-8">#</th>
                  <th className="text-left px-4 py-3">Title</th>
                  <th className="text-left px-4 py-3 hidden md:table-cell">Genres</th>
                  <th className="text-left px-4 py-3 w-48">Predicted rating</th>
                </tr>
              </thead>
              <tbody>
                {recs.map((rec, i) => (
                  <tr
                    key={rec.movieId}
                    className="border-t border-[var(--border)] hover:bg-[var(--card)]/50 transition-colors"
                  >
                    <td className="px-4 py-3 text-[var(--muted)] tabular-nums">
                      {i + 1}
                    </td>
                    <td className="px-4 py-3 font-medium">{rec.title}</td>
                    <td className="px-4 py-3 hidden md:table-cell">
                      <GenreBadges genres={rec.genres} />
                    </td>
                    <td className="px-4 py-3">
                      <RatingBar rating={rec.est_rating} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-[var(--muted)]">
            Avg predicted: {(recs.reduce((s, r) => s + r.est_rating, 0) / recs.length).toFixed(2)} ★
            {" · "}Range: {Math.min(...recs.map((r) => r.est_rating)).toFixed(2)} –{" "}
            {Math.max(...recs.map((r) => r.est_rating)).toFixed(2)}
          </p>
        </div>
      )}

      {recs && recs.length === 0 && (
        <p className="text-sm text-[var(--muted)]">
          This user was not seen during training. No recommendations available.
        </p>
      )}
    </div>
  );
}
