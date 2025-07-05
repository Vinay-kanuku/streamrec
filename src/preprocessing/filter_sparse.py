import os

import polars as pl

from config import (
    CLEAN_RATINGS_FILE,
    FILTER_DIR,
    FILTERED_RATINGS_FILE,
    MIN_MOVIE_RATINGS,
    MIN_USER_RATINGS,
)


def filter_sparse_users_movies():

    ratings = pl.read_csv(CLEAN_RATINGS_FILE)

    # Count how many ratings each user has
    user_counts = ratings.group_by("userId").agg(
        pl.count("rating").alias("user_rating_count")
    )

    # Count how many ratings each movie has
    movie_counts = ratings.group_by("movieId").agg(
        pl.count("rating").alias("movie_rating_count")
    )

    ratings = ratings.join(user_counts, on="userId", how="inner").join(
        movie_counts, on="movieId", how="inner"
    )

    # print(ratings   )

    filtered = ratings.filter(
        (pl.col("user_rating_count") >= MIN_USER_RATINGS)
        & (pl.col("movie_rating_count") >= MIN_MOVIE_RATINGS)
    )

    filtered = filtered.drop(["user_rating_count", "movie_rating_count"])
    # print(filtered)
    try:
        os.makedirs(FILTER_DIR, exist_ok=True)
        filtered.write_csv(FILTERED_RATINGS_FILE)
    except (FileNotFoundError, FileExistsError) as e:
        raise (e)
    print(f"Filtered sparse movies are sotres {FILTERED_RATINGS_FILE}")

    return filtered


if __name__ == "__main__":
    filter_sparse_users_movies()
