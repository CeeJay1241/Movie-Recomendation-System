"""
main.py
Query-based CLI interface for the Movie Recommendation System.

Usage examples:
    python main.py --setup
    python main.py --recommend 42
    python main.py --evaluate
    python main.py --history 42
    python main.py --genre Action
    python main.py --top-rated
    python main.py --predict 42 1

All recommendation and query results are backed by the SQLite database
(data/movies.db) and can be retrieved with plain SQL via src/database.py.
"""

import argparse
import sys
from src.recommender import MovieRecommender


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="movie-recommender",
        description="Personalized movie recommendation system (SVD collaborative filtering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                 Download data, train model, init DB
  python main.py --recommend 42          Top-10 recommendations for user 42
  python main.py --recommend 42 --n 5   Top-5 recommendations
  python main.py --evaluate              Print RMSE, MAE, Precision@10, Recall@10
  python main.py --history 42            Show user 42's past ratings
  python main.py --genre "Sci-Fi"        Search movies by genre
  python main.py --top-rated             Show globally top-rated movies
  python main.py --predict 42 1          Predict user 42's rating for movie 1
  python main.py --sql "SELECT * FROM movies LIMIT 5"
        """,
    )

    p.add_argument("--setup",      action="store_true", help="Download data and train model")
    p.add_argument("--retrain",    action="store_true", help="Force model retraining")
    p.add_argument("--evaluate",   action="store_true", help="Evaluate model performance")
    p.add_argument("--recommend",  type=int, metavar="USER_ID", help="Get recommendations for a user")
    p.add_argument("--n",          type=int, default=10, metavar="N", help="Number of recommendations (default: 10)")
    p.add_argument("--history",    type=int, metavar="USER_ID", help="Show a user's rating history")
    p.add_argument("--genre",      type=str, metavar="GENRE", help="Search movies by genre")
    p.add_argument("--top-rated",  action="store_true", help="Show globally top-rated movies")
    p.add_argument("--predict",    nargs=2, type=int, metavar=("USER_ID", "MOVIE_ID"),
                   help="Predict a user's rating for a specific movie")
    p.add_argument("--sql",        type=str, metavar="QUERY", help="Run a raw SQL query against the database")
    return p


def fmt_df(df, max_rows: int = 20) -> str:
    """Format a DataFrame for terminal display."""
    import pandas as pd
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 120)
    if len(df) > max_rows:
        return df.head(max_rows).to_string(index=False) + f"\n  ... ({len(df) - max_rows} more rows)"
    return df.to_string(index=False)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Default: show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    rec = MovieRecommender()

    # --setup: always run first
    if args.setup or args.retrain:
        rec.setup(force_retrain=args.retrain)

    # For all other commands, setup is needed (lazy-load data + model)
    needs_setup = any([
        args.evaluate,
        args.recommend is not None,
        args.history is not None,
        args.genre is not None,
        args.top_rated,
        args.predict is not None,
    ])

    if needs_setup and not (args.setup or args.retrain):
        rec.setup()

    # --evaluate
    if args.evaluate:
        rec.evaluate()

    # --recommend USER_ID
    if args.recommend is not None:
        print(f"\nTop-{args.n} recommendations for user {args.recommend}:")
        recs = rec.recommend(user_id=args.recommend, n=args.n)
        print(fmt_df(recs))

    # --history USER_ID
    if args.history is not None:
        print(f"\nRating history for user {args.history}:")
        hist = rec.get_user_history(args.history)
        if hist.empty:
            print(f"  No ratings found for user {args.history}.")
        else:
            print(fmt_df(hist))

    # --genre GENRE
    if args.genre is not None:
        print(f"\nMovies in genre '{args.genre}':")
        results = rec.search_by_genre(args.genre)
        if results.empty:
            print(f"  No movies found for genre '{args.genre}'.")
        else:
            print(fmt_df(results))

    # --top-rated
    if args.top_rated:
        print("\nGlobally top-rated movies (min. 50 ratings):")
        top = rec.top_rated()
        print(fmt_df(top))

    # --predict USER_ID MOVIE_ID
    if args.predict is not None:
        user_id, movie_id = args.predict
        est = rec.predict(user_id, movie_id)
        from src.database import get_movie_by_id
        try:
            movie = get_movie_by_id(movie_id)
            print(f"\nPredicted rating: {est} / 5.0")
            print(f"  User   : {user_id}")
            print(f"  Movie  : {movie['title']} (id={movie_id})")
        except KeyError:
            print(f"\nPredicted rating: {est} / 5.0  (movie id={movie_id})")

    # --sql QUERY
    if args.sql is not None:
        from src.database import query
        print(f"\nSQL: {args.sql}")
        try:
            result = query(args.sql)
            print(fmt_df(result))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
