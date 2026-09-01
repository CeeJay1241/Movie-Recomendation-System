"use client";

import { useState, useEffect } from "react";
import { StarRating } from "@/components/star-rating";
import { GenreBadges } from "@/components/genre-badge";

const GENRES = [
  "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
];

interface Movie {
  movieId: number;
  title: string;
  genres: string;
}

interface TopRated {
  movieId: number;
  title: string;
  genres: string;
  avg_rating: number;
  num_ratings: number;
}

export default function BrowsePage() {
  const [genre, setGenre] = useState("Action");
  const [genreMovies, setGenreMovies] = useState<Movie[]>([]);
  const [topRated, setTopRated] = useState<TopRated[]>([]);
  const [genreData, setGenreData] = useState<Record<string, Movie[]> | null>(null);

  useEffect(() => {
    fetch("/data/genres.json")
      .then((r) => r.json())
      .then((data) => {
        setGenreData(data);
        setGenreMovies(data["Action"] || []);
      });
    fetch("/data/top_rated.json")
      .then((r) => r.json())
      .then(setTopRated);
  }, []);

  useEffect(() => {
    if (genreData) {
      setGenreMovies(genreData[genre] || []);
    }
  }, [genre, genreData]);

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-bold tracking-tight mb-2">Browse movies</h1>
        <p className="text-sm text-[var(--muted)]">
          Search by genre or see the all-time community favorites.
        </p>
      </section>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Genre search */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">By genre</h2>
          <div className="flex flex-wrap gap-2">
            {GENRES.map((g) => (
              <button
                key={g}
                onClick={() => setGenre(g)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  genre === g
                    ? "bg-[var(--accent)] text-white"
                    : "bg-[var(--card)] text-[var(--muted)] hover:text-[var(--foreground)] border border-[var(--border)]"
                }`}
              >
                {g}
              </button>
            ))}
          </div>
          <p className="text-xs text-[var(--muted)]">
            {genreMovies.length.toLocaleString()} movies in {genre}
          </p>
          <div className="border border-[var(--border)] rounded-lg overflow-hidden max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0">
                <tr className="bg-[var(--card)] text-[var(--muted)] text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-2">Title</th>
                  <th className="text-left px-4 py-2">Genres</th>
                </tr>
              </thead>
              <tbody>
                {genreMovies.slice(0, 100).map((m) => (
                  <tr
                    key={m.movieId}
                    className="border-t border-[var(--border)]"
                  >
                    <td className="px-4 py-2">{m.title}</td>
                    <td className="px-4 py-2">
                      <GenreBadges genres={m.genres} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {genreMovies.length > 100 && (
              <p className="text-xs text-[var(--muted)] text-center py-2">
                Showing first 100 of {genreMovies.length.toLocaleString()}
              </p>
            )}
          </div>
        </div>

        {/* Top rated */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">All-time top rated</h2>
          <p className="text-xs text-[var(--muted)]">
            Minimum 50 ratings, sorted by average community score
          </p>
          <div className="border border-[var(--border)] rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[var(--card)] text-[var(--muted)] text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-2 w-8">#</th>
                  <th className="text-left px-4 py-2">Title</th>
                  <th className="text-left px-4 py-2 w-32">Rating</th>
                  <th className="text-right px-4 py-2 w-16">Votes</th>
                </tr>
              </thead>
              <tbody>
                {topRated.map((m, i) => (
                  <tr
                    key={m.movieId}
                    className="border-t border-[var(--border)]"
                  >
                    <td className="px-4 py-2 text-[var(--muted)] tabular-nums">
                      {i + 1}
                    </td>
                    <td className="px-4 py-2 font-medium">{m.title}</td>
                    <td className="px-4 py-2">
                      <StarRating rating={m.avg_rating} />
                    </td>
                    <td className="px-4 py-2 text-right text-[var(--muted)] tabular-nums">
                      {m.num_ratings}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
