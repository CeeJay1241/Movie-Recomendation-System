"""
recommender.py
High-level Recommender class that orchestrates data loading,
model training, evaluation, and recommendation retrieval.
"""

import pandas as pd
from src import data_loader, database, model as model_module, evaluate


class MovieRecommender:
    """
    End-to-end movie recommendation system using SVD collaborative filtering.

    Usage:
        rec = MovieRecommender()
        rec.setup()                      # download data, train model, init DB
        recs = rec.recommend(user_id=42) # top-10 recs for user 42
        rec.evaluate()                   # print evaluation metrics
    """

    def __init__(self, n_components: int = 100, n_recs: int = 10):
        self.n_components = n_components
        self.n_recs = n_recs
        self.model: model_module.SVDRecommender = None
        self.test_df: pd.DataFrame = None
        self.movies: pd.DataFrame = None
        self.ratings: pd.DataFrame = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, force_retrain: bool = False) -> None:
        """
        Full setup pipeline:
          1. Download + preprocess MovieLens data
          2. Train SVD model (or load cached model)
          3. Initialise SQLite database
        """
        print("Loading data...")
        self.movies, self.ratings, ratings_norm = data_loader.load_all()
        print(
            f"  {len(self.movies)} movies | "
            f"{len(self.ratings)} ratings | "
            f"{self.ratings['userId'].nunique()} users"
        )

        try:
            if force_retrain:
                raise FileNotFoundError("Forcing retrain.")
            self.model = model_module.load_model()
            print("Loaded cached model.")
            # Rebuild a test split for evaluation (same seed → same split)
            _, _, self.test_df = model_module.train_model(
                self.ratings, n_components=self.n_components
            )
        except FileNotFoundError:
            print(f"Training SVD model (n_components={self.n_components})...")
            self.model, _, self.test_df = model_module.train_model(
                self.ratings, n_components=self.n_components
            )
            model_module.save_model(self.model)
            print(
                f"  Explained variance: "
                f"{self.model.explained_variance_ratio:.1%}"
            )

        print("Initialising database...")
        database.init_db(self.movies, ratings_norm)
        print("Setup complete.\n")

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, n: int = None) -> pd.DataFrame:
        """
        Return top-N recommendations for a user as a DataFrame.
        Columns: movieId, title, genres, est_rating
        """
        self._require_setup()
        n = n or self.n_recs

        rated_ids = set(
            self.ratings[self.ratings["userId"] == user_id]["movieId"]
        )

        raw_recs = self.model.get_top_n_recommendations(
            user_id=user_id, rated_movie_ids=rated_ids, n=n
        )

        rec_df = pd.DataFrame(raw_recs)
        rec_df = rec_df.merge(
            self.movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
        rec_df["genres"] = rec_df["genres"].apply(
            lambda g: "|".join(g) if isinstance(g, list) else g
        )
        rec_df = rec_df[["movieId", "title", "genres", "est_rating"]]
        rec_df["est_rating"] = rec_df["est_rating"].round(3)

        database.store_recommendations(user_id, rec_df)
        return rec_df

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, k: int = 10) -> dict:
        """Evaluate the model and print a report. Returns the metrics dict."""
        self._require_setup()
        if self.test_df is None:
            raise RuntimeError("No test set available. Run setup() first.")
        metrics = evaluate.evaluate_model(self.model, self.test_df, k=k)
        evaluate.print_evaluation_report(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Convenience query methods (delegate to database layer)
    # ------------------------------------------------------------------

    def get_user_history(self, user_id: int) -> pd.DataFrame:
        return database.get_user_ratings(user_id)

    def search_by_genre(self, genre: str) -> pd.DataFrame:
        return database.get_movies_by_genre(genre)

    def top_rated(self, n: int = 10, min_ratings: int = 50) -> pd.DataFrame:
        return database.get_top_rated_movies(n=n, min_ratings=min_ratings)

    def predict(self, user_id: int, movie_id: int) -> float:
        self._require_setup()
        return self.model.predict(user_id, movie_id)

    # ------------------------------------------------------------------

    def _require_setup(self):
        if self.model is None or self.movies is None:
            raise RuntimeError("Call setup() before using the recommender.")
