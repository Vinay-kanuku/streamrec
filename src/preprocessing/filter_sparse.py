import polars as pl
from src.config import (
    CLEAN_RATINGS_FILE, FILTERED_RATINGS_FILE,
    FILTER_DIR, 
    MIN_USER_RATINGS, MIN_MOVIE_RATINGS
    )
import os 

def filter_sparse_users_movies():


    ratings = pl.read_csv(CLEAN_RATINGS_FILE)

    # Correct: group_by (Polars)
    # Count how many ratings each user has
    user_counts = (
        ratings.group_by("userId")
            .agg(pl.count("rating").alias("user_rating_count"))
    )

    # Count how many ratings each movie has
    movie_counts = (
        ratings.group_by("movieId")
            .agg(pl.count("rating").alias("movie_rating_count"))
    )


    ratings = (
        ratings.join(user_counts, on="userId", how="inner")
               .join(movie_counts, on="movieId", how="inner")
    )

    filtered = ratings.filter(
        (pl.col("user_rating_count") >= MIN_USER_RATINGS) &
        (pl.col("movie_rating_count") >= MIN_MOVIE_RATINGS)
    )

    filtered = filtered.drop(["user_rating_count", "movie_rating_count"])
    os.makedirs(FILTER_DIR)
    filtered_file = os.path.join(FILTER_DIR, FILTERED_RATINGS_FILE)
    filtered.write_csv(filtered_file)

    print(f"✅ Filtered ratings saved to: {FILTERED_RATINGS_FILE}")
    print(f"📊 Remaining users: {filtered['userId'].n_unique()}, movies: {filtered['movieId'].n_unique()}")
