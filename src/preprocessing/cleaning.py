import os

import pandas as pd

from src.config import (
    CLEAN_DIR,
    CLEAN_MOVIES_FILE,
    CLEAN_RATINGS_FILE,
    MOVIES_FILE,
    RATINGS_FILE,
)


def clean_movielens_data():
    """Clean MovieLens data and save to clean directory"""
    try:
        ratings = pd.read_csv(RATINGS_FILE)
        movies = pd.read_csv(MOVIES_FILE)
        ratings.drop_duplicates()
        movies.drop_duplicates()
    except FileExistsError as e:
        raise (e)
    except FileNotFoundError as e:
        raise (e)

    try:
        os.makedirs(CLEAN_DIR, exist_ok=True)
        clean_ratings_path = os.path.join(CLEAN_DIR, CLEAN_RATINGS_FILE)
        clean_movies_path = os.path.join(CLEAN_DIR, CLEAN_MOVIES_FILE)
    except Exception as e:
        raise (e)

    ratings.to_csv(clean_ratings_path, index=False)
    movies.to_csv(clean_movies_path, index=False)

    return ratings, movies


if __name__ == "__main__":
    clean_movielens_data()
