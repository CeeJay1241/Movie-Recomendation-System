# Movie Recommendation System

A personalized movie recommendation system using **collaborative filtering** (SVD matrix factorization), backed by a **SQLite** database and evaluated with **precision** and **recall** metrics.

**Stack:** Python · Pandas · Scikit-learn · Surprise · SQLAlchemy · SQLite

---

## How It Works

1. **Data** — Downloads the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) dataset (~100k ratings, 9k movies, 600 users).
2. **Preprocessing** — Ratings are mean-centered per user to reduce bias from users who rate everything high or low.
3. **Model** — An SVD model (via the Surprise library) factorizes the user-item rating matrix into latent factor vectors that capture hidden preferences.
4. **Database** — Movies, ratings, and recommendations are stored in a SQLite database queried through SQLAlchemy.
5. **Evaluation** — Performance is measured with RMSE, MAE, Precision@K, and Recall@K.

---

## Setup

```bash
pip install -r requirements.txt
python main.py --setup
```

This downloads the dataset (~3 MB), trains the model, and initialises the database. Takes ~30 seconds on first run.

---

## Usage

```bash
# Get top-10 recommendations for user 42
python main.py --recommend 42

# Get top-5 recommendations
python main.py --recommend 42 --n 5

# Evaluate model performance (RMSE, MAE, Precision@10, Recall@10)
python main.py --evaluate

# Show a user's past ratings
python main.py --history 42

# Search movies by genre
python main.py --genre "Sci-Fi"

# Show globally top-rated movies
python main.py --top-rated

# Predict a specific user's rating for a specific movie
python main.py --predict 42 1

# Run a raw SQL query against the database
python main.py --sql "SELECT title, avg_rating FROM movies JOIN ..."
```

---

## Project Structure

```
recom/
├── main.py               # CLI entry point
├── requirements.txt
├── data/
│   ├── ml-latest-small/  # MovieLens data (auto-downloaded)
│   ├── movies.db         # SQLite database
│   └── svd_model.pkl     # Serialized trained model
└── src/
    ├── data_loader.py    # Download, preprocess, normalize ratings
    ├── database.py       # SQLite schema, bulk load, query helpers
    ├── model.py          # SVD training, prediction, serialization
    ├── evaluate.py       # RMSE, MAE, Precision@K, Recall@K
    └── recommender.py    # High-level orchestration class
```

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| SVD over user-based CF | Scales to large sparse matrices; handles cold-start better |
| Mean-centering ratings | Removes per-user rating bias before computing cosine similarities |
| SQLite + SQLAlchemy | Portable, zero-config database; parameterized queries prevent SQL injection |
| Surprise library | Battle-tested CF algorithms with built-in cross-validation support |
| Precision@K / Recall@K | More informative than RMSE alone for ranking quality |

---

## Example Output

```
Top-10 recommendations for user 42:
 movieId                               title           genres  est_rating
    1196  Star Wars: Ep. V - Empire Strikes Back  Action|Adventure|Sci-Fi       4.521
     260        Star Wars: Ep. IV - A New Hope  Action|Adventure|Sci-Fi       4.498
     ...

========================================
  MODEL EVALUATION REPORT
========================================
  RMSE              : 0.8731
  MAE               : 0.6704
  Precision@10      : 0.3812
  Recall@10         : 0.2147
  (liked threshold  : ≥ 3.5 stars)
========================================
```
