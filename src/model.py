"""
model.py
Collaborative filtering via SVD matrix factorization using scikit-learn.

How it works:
  1. Build a user-item rating matrix (users × movies).
  2. Mean-center each row (remove per-user rating bias).
  3. Apply TruncatedSVD to decompose the matrix into latent factors:
         M ≈ U · Σ · Vᵀ
     where U captures user preferences and Vᵀ captures item attributes.
  4. Reconstruct the full matrix to obtain predicted ratings for every
     (user, movie) pair, including unseen ones.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "svd_model.pkl")


class SVDRecommender:
    """
    SVD-based collaborative filter.

    Attributes (available after fit()):
        predictions:  ndarray (n_users, n_movies) of reconstructed ratings.
        user_index:   dict mapping userId  → row index.
        item_index:   dict mapping movieId → column index.
        user_ids:     list of userIds  (row order).
        movie_ids:    list of movieIds (column order).
        user_means:   Series of per-user mean ratings.
        global_mean:  float, fallback for unseen users/items.
    """

    def __init__(self, n_components: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.predictions: np.ndarray = None
        self.user_index: dict = {}
        self.item_index: dict = {}
        self.user_ids: list = []
        self.movie_ids: list = []
        self.user_means: pd.Series = None
        self.global_mean: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, ratings: pd.DataFrame) -> "SVDRecommender":
        """
        Train the SVD model on a ratings DataFrame.

        Args:
            ratings: DataFrame with columns [userId, movieId, rating].
        Returns:
            self (for method chaining).
        """
        self.global_mean = ratings["rating"].mean()
        self.user_means = ratings.groupby("userId")["rating"].mean()

        # Build user-item matrix; missing entries are NaN
        matrix = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )

        # Store index mappings so we can look up users/movies by ID later
        self.user_ids = list(matrix.index)
        self.movie_ids = list(matrix.columns)
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_index = {mid: j for j, mid in enumerate(self.movie_ids)}

        # Mean-center each user's ratings, fill remaining NaNs with 0
        user_means_vec = np.array(
            [self.user_means.get(uid, self.global_mean) for uid in self.user_ids]
        )
        matrix_centered = matrix.values - user_means_vec[:, np.newaxis]
        matrix_centered = np.nan_to_num(matrix_centered, nan=0.0)

        # Factorize: fit_transform returns U·Σ  (shape: n_users × n_components)
        U_sigma = self._svd.fit_transform(matrix_centered)  # (n_users, k)
        Vt = self._svd.components_                          # (k, n_movies)

        # Reconstruct and add user means back
        reconstructed = (U_sigma @ Vt) + user_means_vec[:, np.newaxis]

        # Clip to valid rating range
        self.predictions = np.clip(reconstructed, 0.5, 5.0)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating a user would give to a movie.

        Falls back to the user's mean rating for unseen users/movies.
        """
        if user_id not in self.user_index or movie_id not in self.item_index:
            return round(float(self.user_means.get(user_id, self.global_mean)), 3)
        u = self.user_index[user_id]
        m = self.item_index[movie_id]
        return round(float(self.predictions[u, m]), 3)

    def get_top_n_recommendations(
        self,
        user_id: int,
        rated_movie_ids: set,
        n: int = 10,
    ) -> list[dict]:
        """
        Return top-N unseen movie recommendations for a user.

        Args:
            user_id:         Target user.
            rated_movie_ids: Movies already rated — excluded from results.
            n:               Number of recommendations.

        Returns:
            List of dicts [{movieId, est_rating}, ...] sorted descending.
        """
        if user_id not in self.user_index:
            return []
        u = self.user_index[user_id]
        scores = [
            {"movieId": mid, "est_rating": float(self.predictions[u, j])}
            for j, mid in enumerate(self.movie_ids)
            if mid not in rated_movie_ids
        ]
        scores.sort(key=lambda x: x["est_rating"], reverse=True)
        return scores[:n]

    # ------------------------------------------------------------------
    # Explained variance (informational)
    # ------------------------------------------------------------------

    @property
    def explained_variance_ratio(self) -> float:
        """Fraction of variance captured by the latent factors."""
        return round(float(self._svd.explained_variance_ratio_.sum()), 4)


# ---------------------------------------------------------------------------
# Train / save / load helpers
# ---------------------------------------------------------------------------

def train_model(
    ratings: pd.DataFrame,
    n_components: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split ratings into train/test, fit SVDRecommender on train set.

    Returns:
        (model, train_df, test_df)
    """
    train_df, test_df = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    model = SVDRecommender(n_components=n_components, random_state=random_state)
    model.fit(train_df)
    return model, train_df, test_df


def save_model(model: SVDRecommender) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")


def load_model() -> SVDRecommender:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model at {MODEL_PATH}. Run `python main.py --setup` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
