"""
app.py
Streamlit web UI for the Movie Recommendation System.

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.recommender import MovieRecommender

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .genre-badge {
        display: inline-block;
        padding: 2px 10px;
        margin: 2px;
        border-radius: 12px;
        background-color: #1f3a5f;
        color: #e8f0fe;
        font-size: 0.78em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading recommendation model…")
def load_recommender() -> MovieRecommender:
    """
    Instantiate and call setup() ONCE for the entire app session.
    @st.cache_resource keeps the same object in memory across all reruns —
    no re-serialization, no re-training.
    """
    rec = MovieRecommender()
    rec.setup(force_retrain=False)
    return rec


@st.cache_data(show_spinner="Computing evaluation metrics (runs once)…")
def get_cached_metrics(_rec: MovieRecommender) -> dict:
    """
    Cache the slow evaluate() call separately.
    The leading underscore on _rec tells Streamlit to skip hashing it
    (MovieRecommender contains large numpy arrays that are not safely hashable).
    """
    return _rec.evaluate(k=10)


@st.cache_data(show_spinner=False)
def get_cached_top_rated(_rec: MovieRecommender, n: int = 20) -> pd.DataFrame:
    """Cache the top-rated DB query so it doesn't re-run on every tab switch."""
    return _rec.top_rated(n=n, min_ratings=50)


# ---------------------------------------------------------------------------
# Formatter helpers
# ---------------------------------------------------------------------------

def rating_to_stars(r: float) -> str:
    full = int(round(r))
    full = max(0, min(5, full))
    return "★" * full + "☆" * (5 - full) + f"  {r:.2f}"


def format_recommendations_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Stars"] = out["est_rating"].apply(rating_to_stars)
    return out.rename(columns={
        "title": "Title",
        "genres": "Genres",
        "est_rating": "Est. Rating",
    })[["Title", "Genres", "Est. Rating", "Stars"]]


def format_history_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Stars"] = out["rating"].apply(rating_to_stars)
    out["Bias"] = out["rating_norm"].round(2)
    return out.rename(columns={"title": "Title", "rating": "Rating"})[
        ["Title", "Rating", "Stars", "Bias"]
    ]


def analyze_taste_profile(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the pipe-separated genres column so each (movie, genre) pair
    becomes its own row, then aggregate avg rating and watch count per genre.

    Returns a DataFrame with columns: genre, avg_rating, count
    sorted by avg_rating descending.
    Returns empty DataFrame if genres column is absent (stale session state).
    """
    if "genres" not in hist.columns:
        return pd.DataFrame(columns=["genre", "avg_rating", "count"])

    rows = []
    for _, row in hist.iterrows():
        if not row["genres"] or pd.isna(row["genres"]):
            continue
        for genre in row["genres"].split("|"):
            if genre and genre != "(no genres listed)":
                rows.append({"genre": genre, "rating": row["rating"]})

    if not rows:
        return pd.DataFrame(columns=["genre", "avg_rating", "count"])

    genre_df = pd.DataFrame(rows)
    stats = (
        genre_df.groupby("genre")["rating"]
        .agg(avg_rating="mean", count="count")
        .reset_index()
        .sort_values("avg_rating", ascending=False)
    )
    stats["avg_rating"] = stats["avg_rating"].round(2)
    return stats


def format_top_rated_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Stars"] = out["avg_rating"].apply(rating_to_stars)
    return out.rename(columns={
        "title": "Title",
        "genres": "Genres",
        "avg_rating": "Avg Rating",
        "num_ratings": "# Ratings",
    })[["Title", "Genres", "Avg Rating", "# Ratings", "Stars"]]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(rec: MovieRecommender) -> None:
    with st.sidebar:
        st.title("🎬 CineMatch")
        st.caption(
            "Personalized movie recommendations powered by "
            "SVD collaborative filtering"
        )
        st.divider()

        st.success("Model Loaded", icon="✅")
        st.caption(f"SVD · {rec.model.n_components} latent factors")
        st.caption(
            f"Explained variance: {rec.model.explained_variance_ratio:.1%}"
        )
        st.divider()

        st.subheader("User ID")
        st.number_input(
            "Select a user (1 – 610)",
            min_value=1, max_value=610, value=42, step=1,
            key="global_user_id",
            help="This ID is shared across Recommendations and User History tabs.",
        )
        st.divider()

        c1, c2 = st.columns(2)
        c1.metric("Movies", f"{len(rec.movies):,}")
        c2.metric("Users", f"{rec.ratings['userId'].nunique():,}")
        st.metric("Ratings", f"{len(rec.ratings):,}")
        st.divider()

        with st.expander("How to use"):
            st.markdown(
                """
                1. Set your **User ID** in the sidebar
                2. Go to **Recommendations** and click **Get Recommendations**
                3. Go to **User History** and click **Load History**

                Use **Browse Movies** to explore by genre or
                find all-time top-rated films.

                Open **Model Performance** for evaluation metrics.
                """
            )


# ---------------------------------------------------------------------------
# Tab 1 — Recommendations
# ---------------------------------------------------------------------------

def render_recommendations_tab(rec: MovieRecommender) -> None:
    st.header("Personalized Recommendations")
    st.caption(
        "SVD decomposes the user-item rating matrix into latent factors that "
        "capture hidden taste patterns — without any explicit movie features."
    )

    col_input, col_explain = st.columns([1, 2])

    with col_input:
        user_id = st.session_state.get("global_user_id", 42)
        st.info(f"User ID: **{user_id}** — change it in the sidebar")
        n_recs = st.slider(
            "Number of recommendations", min_value=1, max_value=20, value=10,
            key="rec_n",
        )
        get_btn = st.button(
            "Get Recommendations", type="primary", use_container_width=True
        )

    with col_explain:
        st.info(
            """
            **How it works**

            The model factorizes the 610 × 9 000 user-item rating matrix into
            100 latent dimensions using Truncated SVD (scikit-learn).
            For each user it predicts ratings for every **unrated** movie and
            returns the top-N by predicted score.
            """
        )

    st.divider()

    if get_btn:
        with st.spinner(f"Finding top {n_recs} picks for user {user_id}…"):
            recs = rec.recommend(user_id=int(user_id), n=n_recs)
        st.session_state["last_recs"] = recs
        st.session_state["last_user"] = int(user_id)

    if "last_recs" in st.session_state:
        recs = st.session_state["last_recs"]
        shown_user = st.session_state.get("last_user", user_id)

        if recs.empty:
            st.warning(
                f"User {shown_user} was not seen during training. "
                "No recommendations available."
            )
        else:
            st.subheader(f"Top {len(recs)} picks for User {shown_user}")
            display = format_recommendations_df(recs)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Est. Rating": st.column_config.ProgressColumn(
                        "Est. Rating",
                        min_value=0.5,
                        max_value=5.0,
                        format="%.2f ★",
                    ),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Genres": st.column_config.TextColumn("Genres", width="medium"),
                    "Stars": st.column_config.TextColumn("Stars", width="medium"),
                },
            )
            st.caption(
                f"Avg predicted rating: {recs['est_rating'].mean():.2f}  ·  "
                f"Range: {recs['est_rating'].min():.2f} – "
                f"{recs['est_rating'].max():.2f}"
            )
            st.download_button(
                label="⬇ Download as CSV",
                data=recs.to_csv(index=False),
                file_name=f"recommendations_user{shown_user}.csv",
                mime="text/csv",
            )
    else:
        st.markdown(
            "Enter a User ID above and click **Get Recommendations** to see results."
        )


# ---------------------------------------------------------------------------
# Tab 2 — Model Performance
# ---------------------------------------------------------------------------

def render_performance_tab(rec: MovieRecommender) -> None:
    st.header("Model Performance")
    st.caption(
        "Evaluated on a held-out 20% test split (random seed 42). "
        "Results are cached after the first computation."
    )

    metrics = get_cached_metrics(rec)

    st.divider()

    # --- 4 metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{metrics['rmse']:.4f}",
              help="Root Mean Squared Error — average star-rating error. Lower is better.")
    c2.metric("MAE", f"{metrics['mae']:.4f}",
              help="Mean Absolute Error — less sensitive to outliers than RMSE. Lower is better.")
    c3.metric(f"Precision@{metrics['k']}", f"{metrics['precision_at_k']:.4f}",
              help=f"Of the top {metrics['k']} recommendations, fraction the user actually liked. Higher is better.")
    c4.metric(f"Recall@{metrics['k']}", f"{metrics['recall_at_k']:.4f}",
              help=f"Fraction of liked movies captured in the top {metrics['k']}. Higher is better.")

    st.divider()

    # --- Plotly bar chart ---
    names  = ["RMSE", "MAE",
              f"Precision@{metrics['k']}", f"Recall@{metrics['k']}"]
    values = [metrics["rmse"], metrics["mae"],
              metrics["precision_at_k"], metrics["recall_at_k"]]
    colors = ["#ef553b", "#ef553b", "#00cc96", "#00cc96"]

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            width=0.4,
        )
    ])
    fig.update_layout(
        title="Evaluation Metrics",
        yaxis=dict(range=[0, max(values) * 1.5], title="Score"),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        height=380,
        margin=dict(t=50, b=40),
    )
    fig.add_annotation(
        text="red = error metrics (lower is better)   "
             "green = ranking metrics (higher is better)",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, color="grey"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Plain-English explanations ---
    st.subheader("What these metrics mean")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            **RMSE** — On average, predicted ratings are off by
            **{metrics['rmse']:.2f} stars**.
            Lower values mean more accurate rating predictions.

            **MAE** — Similar to RMSE but less sensitive to large individual
            errors. An MAE of {metrics['mae']:.2f} means the typical
            prediction is within {metrics['mae']:.2f} stars of the truth.
            """
        )
    with right:
        st.markdown(
            f"""
            **Precision@{metrics['k']}** — Of the {metrics['k']} movies
            recommended to each user, **{metrics['precision_at_k']:.1%}**
            were ones they actually liked (rated ≥ {metrics['threshold']} ★).

            **Recall@{metrics['k']}** — Of all movies a user liked,
            **{metrics['recall_at_k']:.1%}** appeared in their
            top-{metrics['k']} recommendations.
            """
        )


# ---------------------------------------------------------------------------
# Tab 3 — Browse Movies
# ---------------------------------------------------------------------------

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def render_browse_tab(rec: MovieRecommender) -> None:
    st.header("Browse Movies")

    col_genre, col_top = st.columns(2)

    with col_genre:
        st.subheader("Search by Genre")
        genre = st.selectbox("Select a genre", GENRES)
        results = rec.search_by_genre(genre)

        if results.empty:
            st.info(f"No movies found for genre '{genre}'.")
        else:
            st.caption(f"{len(results)} movies in **{genre}**")
            display = results[["title", "genres"]].rename(
                columns={"title": "Title", "genres": "Genres"}
            )
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                height=390,
                column_config={
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Genres": st.column_config.TextColumn("Genres", width="medium"),
                },
            )
            st.download_button(
                label="⬇ Download as CSV",
                data=display.to_csv(index=False),
                file_name=f"movies_{genre.lower()}.csv",
                mime="text/csv",
                key="dl_genre",
            )

    with col_top:
        st.subheader("All-Time Top Rated")
        st.caption("Minimum 50 ratings · sorted by average community score")
        top_df = get_cached_top_rated(rec, n=20)
        display_top = format_top_rated_df(top_df)
        st.dataframe(
            display_top,
            use_container_width=True,
            hide_index=True,
            height=390,
            column_config={
                "Title": st.column_config.TextColumn("Title", width="large"),
                "Avg Rating": st.column_config.NumberColumn(
                    "Avg Rating", format="%.2f"
                ),
                "# Ratings": st.column_config.NumberColumn(
                    "# Ratings", format="%d"
                ),
                "Stars": st.column_config.TextColumn("Stars", width="medium"),
            },
        )
        st.download_button(
            label="⬇ Download as CSV",
            data=display_top.to_csv(index=False),
            file_name="top_rated_movies.csv",
            mime="text/csv",
            key="dl_top",
        )


# ---------------------------------------------------------------------------
# Tab 4 — User History
# ---------------------------------------------------------------------------

def render_history_tab(rec: MovieRecommender) -> None:
    st.header("User Rating History")
    st.caption("Explore what a specific user has watched and rated.")

    col_in, _ = st.columns([1, 3])
    with col_in:
        hist_uid = st.session_state.get("global_user_id", 42)
        st.info(f"User ID: **{hist_uid}** — change it in the sidebar")
        load_btn = st.button("Load History", type="primary", key="hist_btn")

    if load_btn:
        with st.spinner(f"Loading history for user {hist_uid}…"):
            hist = rec.get_user_history(int(hist_uid))
        st.session_state["history_df"] = hist
        st.session_state["history_user"] = int(hist_uid)

    if "history_df" in st.session_state:
        hist = st.session_state["history_df"]
        shown_uid = st.session_state.get("history_user", hist_uid)

        if hist.empty:
            st.warning(f"No rating history found for user {shown_uid}.")
            return

        # Summary metrics
        best_title = hist.loc[hist["rating"].idxmax(), "title"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Movies Rated", f"{len(hist):,}")
        m2.metric("Avg Rating", f"{hist['rating'].mean():.2f} ★")
        m3.metric("Highest Rated", best_title[:28] + ("…" if len(best_title) > 28 else ""))

        st.divider()

        col_table, col_chart = st.columns([1.2, 1])

        with col_table:
            st.subheader(f"User {shown_uid}'s Ratings")
            display_hist = format_history_df(hist)
            st.dataframe(
                display_hist,
                use_container_width=True,
                hide_index=True,
                height=350,
                column_config={
                    "Rating": st.column_config.NumberColumn("Rating", format="%.1f"),
                    "Bias": st.column_config.NumberColumn(
                        "Bias",
                        help="Mean-centered: positive = rated above their own average",
                        format="%.2f",
                    ),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                },
            )
            st.download_button(
                label="⬇ Download as CSV",
                data=display_hist.to_csv(index=False),
                file_name=f"history_user{shown_uid}.csv",
                mime="text/csv",
                key="dl_hist",
            )

        with col_chart:
            st.subheader("Rating Distribution")
            fig = px.histogram(
                hist,
                x="rating",
                nbins=9,
                range_x=[0.5, 5.5],
                labels={"rating": "Rating (stars)", "count": "Movies"},
                color_discrete_sequence=["#636EFA"],
                title=f"User {shown_uid} — Rating Distribution",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                bargap=0.08,
                yaxis_title="Number of Movies",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Taste Profile ---
        st.divider()
        st.subheader("Taste Profile")

        taste = analyze_taste_profile(hist)

        if "genres" not in hist.columns:
            st.info("Click **Load History** again to refresh and see the Taste Profile.")
        elif not taste.empty:
            # Personality label based on rating bias
            user_avg = hist["rating"].mean()
            dataset_avg = 3.5  # approximate MovieLens global mean
            bias = user_avg - dataset_avg
            if bias > 0.3:
                personality = f"Generous rater — rates {bias:+.2f}★ above average"
                personality_icon = "😊"
            elif bias < -0.3:
                personality = f"Harsh critic — rates {abs(bias):.2f}★ below average"
                personality_icon = "🧐"
            else:
                personality = "Balanced rater — close to the dataset average"
                personality_icon = "⚖️"

            fav_genre     = taste.iloc[0]["genre"]
            fav_avg       = taste.iloc[0]["avg_rating"]
            most_watched  = taste.sort_values("count", ascending=False).iloc[0]["genre"]
            most_watched_n = taste.sort_values("count", ascending=False).iloc[0]["count"]

            p1, p2, p3 = st.columns(3)
            p1.metric("Favorite Genre", fav_genre,
                      help=f"Highest avg rating: {fav_avg:.2f}★")
            p2.metric("Most Watched Genre", most_watched,
                      help=f"{most_watched_n} movies rated")
            p3.metric(f"{personality_icon} Rater Style", personality.split("—")[0].strip(),
                      delta=f"{bias:+.2f}★ vs avg", delta_color="normal")

            st.caption(personality)

            ch1, ch2 = st.columns(2)

            with ch1:
                # Avg rating per genre — sorted bar chart
                fig_avg = px.bar(
                    taste.sort_values("avg_rating"),
                    x="avg_rating",
                    y="genre",
                    orientation="h",
                    title="Avg Rating by Genre",
                    labels={"avg_rating": "Avg Rating", "genre": "Genre"},
                    color="avg_rating",
                    color_continuous_scale="RdYlGn",
                    range_color=[1, 5],
                    text="avg_rating",
                )
                fig_avg.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_avg.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    coloraxis_showscale=False,
                    height=max(300, len(taste) * 28),
                    margin=dict(l=10, r=60, t=40, b=20),
                    xaxis=dict(range=[0, 5.5]),
                )
                st.plotly_chart(fig_avg, use_container_width=True)

            with ch2:
                # Movies watched per genre — count bar chart
                fig_cnt = px.bar(
                    taste.sort_values("count"),
                    x="count",
                    y="genre",
                    orientation="h",
                    title="Movies Watched per Genre",
                    labels={"count": "# Movies", "genre": "Genre"},
                    color="count",
                    color_continuous_scale="Blues",
                    text="count",
                )
                fig_cnt.update_traces(texttemplate="%{text}", textposition="outside")
                fig_cnt.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    coloraxis_showscale=False,
                    height=max(300, len(taste) * 28),
                    margin=dict(l=10, r=60, t=40, b=20),
                )
                st.plotly_chart(fig_cnt, use_container_width=True)
    else:
        st.markdown("Enter a User ID above and click **Load History**.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rec = load_recommender()
    render_sidebar(rec)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Recommendations",
        "📊 Model Performance",
        "🔍 Browse Movies",
        "👤 User History",
    ])

    with tab1:
        render_recommendations_tab(rec)
    with tab2:
        render_performance_tab(rec)
    with tab3:
        render_browse_tab(rec)
    with tab4:
        render_history_tab(rec)


if __name__ == "__main__":
    main()
