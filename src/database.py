"""
database.py
SQLite-backed storage layer for movies and ratings.

Provides:
  - Schema creation and bulk-loading from DataFrames
  - A simple query interface to retrieve movies and recommendations
    via plain SQL, using SQLAlchemy Core (no ORM required).
"""

import os
import pandas as pd
from sqlalchemy import (
    create_engine,
    text,
    Table,
    Column,
    Integer,
    Float,
    String,
    MetaData,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "movies.db")
DB_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

metadata = MetaData()

movies_table = Table(
    "movies",
    metadata,
    Column("movieId", Integer, primary_key=True),
    Column("title", String, nullable=False),
    Column("genres", String),          # stored as pipe-separated string
)

ratings_table = Table(
    "ratings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("userId", Integer, nullable=False),
    Column("movieId", Integer, nullable=False),
    Column("rating", Float, nullable=False),
    Column("rating_norm", Float),      # mean-centered rating
)


def get_engine():
    return create_engine(DB_URL, echo=False)


def init_db(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """
    Create tables and populate them from DataFrames.
    Drops existing data so the DB always reflects the latest load.

    Args:
        movies:  DataFrame with columns [movieId, title, genres (list)]
        ratings: DataFrame with columns [userId, movieId, rating, rating_norm]
    """
    engine = get_engine()
    metadata.drop_all(engine)
    metadata.create_all(engine)

    # Serialize genre lists back to pipe-separated strings for storage
    movies_db = movies.copy()
    movies_db["genres"] = movies_db["genres"].apply(
        lambda g: "|".join(g) if isinstance(g, list) else g
    )

    movies_db[["movieId", "title", "genres"]].to_sql(
        "movies", con=engine, if_exists="append", index=False
    )
    ratings[["userId", "movieId", "rating", "rating_norm"]].to_sql(
        "ratings", con=engine, if_exists="append", index=False
    )
    print(f"Database initialized at {DB_PATH}")
    print(f"  {len(movies_db)} movies, {len(ratings)} ratings loaded.")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query(sql: str, params: dict = None) -> pd.DataFrame:
    """
    Execute a raw SQL SELECT and return results as a DataFrame.

    Example:
        query("SELECT * FROM movies WHERE genres LIKE :genre",
              {"genre": "%Action%"})
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def get_movie_by_id(movie_id: int) -> pd.Series:
    """Return a single movie row by its movieId."""
    df = query("SELECT * FROM movies WHERE movieId = :id", {"id": movie_id})
    if df.empty:
        raise KeyError(f"No movie found with movieId={movie_id}")
    return df.iloc[0]


def get_movies_by_genre(genre: str) -> pd.DataFrame:
    """Return all movies whose genres include the given genre string."""
    return query(
        "SELECT * FROM movies WHERE genres LIKE :genre",
        {"genre": f"%{genre}%"},
    )


def get_user_ratings(user_id: int) -> pd.DataFrame:
    """Return all ratings submitted by a user, joined with movie titles."""
    return query(
        """
        SELECT r.userId, r.movieId, m.title, r.rating, r.rating_norm
        FROM   ratings r
        JOIN   movies  m ON m.movieId = r.movieId
        WHERE  r.userId = :uid
        ORDER  BY r.rating DESC
        """,
        {"uid": user_id},
    )


def get_top_rated_movies(n: int = 20, min_ratings: int = 50) -> pd.DataFrame:
    """
    Return the top-N movies by average rating, requiring at least
    min_ratings votes to filter out obscure titles with few reviews.
    """
    return query(
        """
        SELECT   m.movieId, m.title, m.genres,
                 ROUND(AVG(r.rating), 2) AS avg_rating,
                 COUNT(r.rating)         AS num_ratings
        FROM     movies  m
        JOIN     ratings r ON r.movieId = m.movieId
        GROUP BY m.movieId
        HAVING   num_ratings >= :min_r
        ORDER BY avg_rating DESC
        LIMIT    :n
        """,
        {"min_r": min_ratings, "n": n},
    )


def store_recommendations(user_id: int, recommendations: pd.DataFrame) -> None:
    """
    Persist a set of recommendations for a user into the
    'recommendations' table (created on first call).

    Args:
        user_id:         The target user.
        recommendations: DataFrame with columns [movieId, title, est_rating].
    """
    engine = get_engine()
    rec_df = recommendations.copy()
    rec_df["userId"] = user_id
    rec_df[["userId", "movieId", "title", "est_rating"]].to_sql(
        "recommendations", con=engine, if_exists="replace", index=False
    )


def get_stored_recommendations(user_id: int) -> pd.DataFrame:
    """Retrieve previously stored recommendations for a user."""
    return query(
        "SELECT * FROM recommendations WHERE userId = :uid ORDER BY est_rating DESC",
        {"uid": user_id},
    )
