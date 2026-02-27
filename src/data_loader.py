"""
data_loader.py
Downloads and preprocesses the MovieLens ml-latest-small dataset.
Produces normalized ratings and movie metadata as Pandas DataFrames.
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-latest-small")


def download_dataset() -> None:
    """Download and extract the MovieLens small dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isdir(EXTRACT_DIR):
        return  # already downloaded

    print("Downloading MovieLens ml-latest-small dataset...")
    response = requests.get(MOVIELENS_URL, timeout=60)
    response.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    os.remove(ZIP_PATH)
    print("Download complete.")


def load_movies() -> pd.DataFrame:
    """Load and return the movies DataFrame with parsed genres."""
    download_dataset()
    path = os.path.join(EXTRACT_DIR, "movies.csv")
    movies = pd.read_csv(path)
    # Split the pipe-separated genres into a list
    movies["genres"] = movies["genres"].str.split("|")
    return movies


def load_ratings() -> pd.DataFrame:
    """Load raw ratings and return a cleaned DataFrame."""
    download_dataset()
    path = os.path.join(EXTRACT_DIR, "ratings.csv")
    ratings = pd.read_csv(path)
    # Drop the timestamp column — not needed for recommendations
    ratings = ratings.drop(columns=["timestamp"])
    return ratings


def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ratings using per-user mean centering.

    Each user's ratings are shifted by their personal mean so that
    a score of 0 represents an 'average' rating for that user.
    This reduces bias from users who rate everything high or low.

    Returns a copy of the DataFrame with an added 'rating_norm' column.
    """
    ratings = ratings.copy()
    user_means = ratings.groupby("userId")["rating"].transform("mean")
    ratings["rating_norm"] = ratings["rating"] - user_means
    return ratings


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user-item matrix from the ratings DataFrame.

    Rows = users, columns = movies, values = raw ratings.
    Missing entries are NaN (sparse — most users haven't rated most movies).
    """
    matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    return matrix


def load_all():
    """Convenience function: returns (movies, ratings, ratings_normalized)."""
    movies = load_movies()
    ratings = load_ratings()
    ratings_norm = normalize_ratings(ratings)
    return movies, ratings, ratings_norm
