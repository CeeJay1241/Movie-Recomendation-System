# CineMatch — Movie Recommendation System

A personalized movie recommendation system with an interactive web UI, powered by **SVD collaborative filtering**, a **SQLite** database, and evaluated with **precision** and **recall** metrics.

**Stack:** Python · Pandas · Scikit-learn · SQLAlchemy · SQLite · Streamlit · Plotly

---

## Features

- **Personalized Recommendations** — Enter any user ID and get a ranked list of movies predicted to match their taste, with a visual rating bar for each result
- **Download Results as CSV** — Every table in the app (recommendations, browsing results, user history) has a one-click CSV export button
- **Model Performance Dashboard** — Visual evaluation report with RMSE, MAE, Precision@10, and Recall@10 metrics, displayed as metric cards and an interactive bar chart
- **Browse by Genre** — Filter the full 9,000-movie catalogue by genre (Action, Sci-Fi, Drama, etc.) and see results instantly
- **All-Time Top Rated** — View the highest community-rated movies, filtered to only include films with at least 50 ratings
- **User History Explorer** — Look up any user's full rating history with a Plotly histogram showing how they distribute their scores
- **Taste Profile** — Automatically analyzes a user's genre preferences: favorite genre, most-watched genre, avg rating per genre (color-coded bar chart), and a rater personality label (generous / balanced / harsh critic)
- **Raw SQL Interface** — Query the SQLite database directly from the command line for custom lookups
- **Automatic Data Download** — The MovieLens dataset downloads automatically on first run; no manual setup needed
- **Cached Model Loading** — The trained model is serialized to disk and reloaded instantly on subsequent runs — no retraining required

---

## How It Works

1. **Data** — Downloads the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) dataset (~100k ratings, 9k movies, 600 users) automatically on first run.
2. **Preprocessing** — Ratings are mean-centered per user to remove bias from users who consistently rate everything high or low.
3. **Model** — Truncated SVD (scikit-learn) factorizes the 610 × 9,000 user-item rating matrix into 100 latent dimensions that capture hidden taste patterns without any explicit movie features.
4. **Database** — Movies, ratings, and generated recommendations are stored in a SQLite database and queried through SQLAlchemy with parameterized queries.
5. **Evaluation** — Performance is measured with RMSE, MAE, Precision@K, and Recall@K on a held-out 20% test split.
6. **UI** — A Streamlit web app provides an interactive interface with charts, downloadable tables, and real-time recommendations.

---

## Setup

```bash
pip install -r requirements.txt

# Download dataset, train model, initialize database (~30 seconds first time)
python main.py --setup
```

---

## Launch the Web UI

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`. The model loads from disk instantly after the first run.

---

## CLI Usage

```bash
# Get top-10 recommendations for user 42
python main.py --recommend 42

# Get top-5 recommendations
python main.py --recommend 42 --n 5

# Evaluate model (RMSE, MAE, Precision@10, Recall@10)
python main.py --evaluate

# Show a user's past ratings
python main.py --history 42

# Search movies by genre
python main.py --genre "Sci-Fi"

# Show globally top-rated movies
python main.py --top-rated

# Predict a user's rating for a specific movie
python main.py --predict 42 1

# Run a raw SQL query
python main.py --sql "SELECT title, genres FROM movies WHERE genres LIKE '%Horror%' LIMIT 10"
```

---

## Project Structure

```
recom/
├── app.py                # Streamlit web UI
├── main.py               # CLI entry point
├── requirements.txt
├── data/
│   ├── ml-latest-small/  # MovieLens dataset (auto-downloaded)
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
| SVD (Truncated) via scikit-learn | Scales to large sparse matrices; no C compiler required unlike Surprise |
| Mean-centering ratings per user | Removes systematic rating bias before matrix decomposition |
| SQLite + SQLAlchemy Core | Portable, zero-config database; parameterized queries prevent SQL injection |
| `@st.cache_resource` for model | Keeps 51 MB model object in memory across Streamlit reruns without re-serializing |
| Precision@K / Recall@K | More informative than RMSE alone for ranking-quality evaluation |
| `st.download_button` | Lets users export any result set as a CSV without leaving the browser |

## Images

<img width="1600" height="900" alt="Screenshot (65)" src="https://github.com/user-attachments/assets/04bd45c5-6959-47c0-8441-70ffd042bcd6" />
<img width="1600" height="900" alt="Screenshot (66)" src="https://github.com/user-attachments/assets/4d5c1f4b-136f-4d53-854b-53823e4397bd" />
<img width="1600" height="900" alt="Screenshot (67)" src="https://github.com/user-attachments/assets/4927a6fc-347f-4c9f-b123-7ecf004c4575" />
<img width="1600" height="900" alt="Screenshot (68)" src="https://github.com/user-attachments/assets/194b675e-bd72-4249-8f49-99e63999607b" />
<img width="1600" height="900" alt="Screenshot (69)" src="https://github.com/user-attachments/assets/809336e4-c21f-4027-9dc8-96f198a61a38" />
<img width="1600" height="900" alt="Screenshot (70)" src="https://github.com/user-attachments/assets/5909479b-3d83-4808-9276-8a235e451df8" />
<img width="1600" height="900" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/64afc01f-def3-414c-babc-935c74020b33" />
<img width="1600" height="900" alt="Screenshot (72)" src="https://github.com/user-attachments/assets/03d93229-f913-4a65-a4b3-773652712456" />
<img width="1600" height="900" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/d4599319-e608-4a70-9682-6a099eb51c83" />


