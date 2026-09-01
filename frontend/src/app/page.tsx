import Link from "next/link";
import { MetricCard } from "@/components/metric-card";

async function getData() {
  const summary = (await import("../../public/data/summary.json")).default;
  const metrics = (await import("../../public/data/metrics.json")).default;
  return { summary, metrics };
}

export default async function Home() {
  const { summary, metrics } = await getData();

  return (
    <div className="space-y-12">
      {/* Hero */}
      <section className="pt-12 pb-4">
        <h1 className="text-4xl font-bold tracking-tight mb-3">CineMatch</h1>
        <p className="text-lg text-[var(--muted)] max-w-2xl">
          A movie recommendation engine built on SVD collaborative filtering.
          Decomposes a {summary.total_users.toLocaleString()}-user ×{" "}
          {summary.total_movies.toLocaleString()}-movie rating matrix into{" "}
          {summary.n_components} latent dimensions to predict what you would rate
          a film you have never seen.
        </p>
      </section>

      {/* Stats */}
      <section>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Movies"
            value={summary.total_movies.toLocaleString()}
          />
          <MetricCard
            label="Users"
            value={summary.total_users.toLocaleString()}
          />
          <MetricCard
            label="Ratings"
            value={summary.total_ratings.toLocaleString()}
          />
          <MetricCard
            label="Variance captured"
            value={`${(summary.explained_variance * 100).toFixed(1)}%`}
            subtitle={`${summary.n_components} latent factors`}
          />
        </div>
      </section>

      {/* Model accuracy */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Model accuracy</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="RMSE"
            value={metrics.rmse.toFixed(4)}
            subtitle="Avg star-rating error"
          />
          <MetricCard
            label="MAE"
            value={metrics.mae.toFixed(4)}
            subtitle="Mean absolute error"
          />
          <MetricCard
            label={`Precision@${metrics.k}`}
            value={`${(metrics.precision_at_k * 100).toFixed(1)}%`}
            subtitle="Of top-10, fraction liked"
          />
          <MetricCard
            label={`Recall@${metrics.k}`}
            value={`${(metrics.recall_at_k * 100).toFixed(1)}%`}
            subtitle="Of liked, fraction in top-10"
          />
        </div>
      </section>

      {/* Quick links */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Explore</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {[
            {
              href: "/recommend",
              title: "Get recommendations",
              desc: "Enter a user ID and see their top-10 predicted films.",
            },
            {
              href: "/browse",
              title: "Browse movies",
              desc: "Search by genre or see all-time top-rated films.",
            },
            {
              href: "/user",
              title: "User profiles",
              desc: "Rating history, taste profiles, and rater personality.",
            },
          ].map((card) => (
            <Link
              key={card.href}
              href={card.href}
              className="block bg-[var(--card)] border border-[var(--border)] rounded-lg p-5 hover:border-[var(--accent)] transition-colors group"
            >
              <h3 className="font-medium mb-1 group-hover:text-[var(--accent-light)] transition-colors">
                {card.title}
              </h3>
              <p className="text-sm text-[var(--muted)]">{card.desc}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="pb-8">
        <h2 className="text-lg font-semibold mb-4">How it works</h2>
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-6 space-y-4 text-sm text-[var(--muted)]">
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <p className="text-[var(--foreground)] font-medium mb-2">
                1. Build the matrix
              </p>
              <p>
                Construct a {summary.total_users} × {summary.total_movies.toLocaleString()}{" "}
                user-item rating matrix from {summary.total_ratings.toLocaleString()}{" "}
                ratings. Mean-center each row to remove per-user rating bias.
              </p>
            </div>
            <div>
              <p className="text-[var(--foreground)] font-medium mb-2">
                2. Factorize with SVD
              </p>
              <p>
                Truncated SVD decomposes the matrix: M ≈ U · Σ · Vᵀ. U captures
                user preferences across {summary.n_components} latent dimensions. Vᵀ
                captures item attributes.
              </p>
            </div>
            <div>
              <p className="text-[var(--foreground)] font-medium mb-2">
                3. Reconstruct and predict
              </p>
              <p>
                Multiply back to get predicted ratings for every user-movie pair,
                including unseen ones. Return the highest-predicted unseen films
                as recommendations.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
