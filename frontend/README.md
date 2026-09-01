# CineMatch

A movie recommendation engine built on SVD collaborative filtering. Decomposes a 610-user x 9,742-movie rating matrix into 100 latent dimensions to predict what you would rate a film you have never seen.

Built with Next.js, TypeScript, Tailwind CSS, and Recharts. All recommendation data is pre-computed as static JSON from the [MovieLens](https://grouplens.org/datasets/movielens/) dataset, so the app runs entirely on the edge with zero backend costs.

## Pages

- **Home** — dataset stats, model accuracy, and a visual walkthrough of how SVD collaborative filtering works
- **Recommend** — enter a user ID (1–610) to see their top-10 predicted films with genre badges and rating bars
- **Browse** — filter the catalog by genre or see the all-time community top-rated movies
- **Users** — explore any user's rating history, taste profile charts, and rater personality classification
- **Model** — RMSE, MAE, Precision@10, Recall@10 with an interactive bar chart and plain-English explanations

## Tech stack

- **Model**: TruncatedSVD (scikit-learn) with 100 latent factors on the MovieLens Small dataset (100k ratings)
- **Frontend**: Next.js 16 (App Router), TypeScript, Tailwind CSS
- **Charts**: Recharts
- **Data**: Pre-computed static JSON (610 user profiles, recommendations, taste breakdowns)
- **Hosting**: Vercel (free tier)

## Local development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Deployment

Push to GitHub and import the repository on [vercel.com/new](https://vercel.com/new). No environment variables or build configuration needed — the defaults work out of the box.
