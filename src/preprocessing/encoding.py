import json
import os
import sys

import polars as pl

# Fix import path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    ENCODED_DIR,
    ENCODED_RATINGS_FILE,
    FILTERED_RATINGS_FILE,
    MOVIE2INDEX_FILE,
    USER2INDEX_FILE,
)


def encode_ids():
    print(" [Polars] Encoding userId and movieId...")

    # Load filtered ratings
    df = pl.read_csv(FILTERED_RATINGS_FILE)

    # Create user mapping using row_number (more efficient)
    user_mapping = (
        df.select("userId")
        .unique()
        .sort("userId")
        .with_columns(pl.int_range(pl.len()).alias("user_index"))
    )

    # Create movie mapping using row_number
    movie_mapping = (
        df.select("movieId")
        .unique()
        .sort("movieId")
        .with_columns(pl.int_range(pl.len()).alias("movie_index"))
    )

    # Join mappings back to original dataframe (much faster than apply/map_elements)
    df_encoded = df.join(user_mapping, on="userId", how="left").join(
        movie_mapping, on="movieId", how="left"
    )

    # Save encoded ratings
    os.makedirs(ENCODED_DIR, exist_ok=True)
    encoded_rating_file = os.path.join(ENCODED_DIR, ENCODED_RATINGS_FILE)  # Fixed typo
    df_encoded.write_csv(encoded_rating_file)

    # Convert mappings to dictionaries for saving
    user2index = dict(
        zip(user_mapping["userId"].to_list(), user_mapping["user_index"].to_list())
    )

    movie2index = dict(
        zip(movie_mapping["movieId"].to_list(), movie_mapping["movie_index"].to_list())
    )

    # Save mappings as JSON
    with open(USER2INDEX_FILE, "w") as f:
        json.dump(user2index, f)

    with open(MOVIE2INDEX_FILE, "w") as f:
        json.dump(movie2index, f)

    print(f" Encoded ratings → {ENCODED_RATINGS_FILE}")
    print(f" user2index → {USER2INDEX_FILE} ({len(user2index)} users)")
    print(f" movie2index → {MOVIE2INDEX_FILE} ({len(movie2index)} movies)")

    return df_encoded


def encode_ids_categorical():
    """
    Ultra-fast version using Polars categorical encoding.
    This is the most efficient approach for large datasets.
    """
    print(" [Polars] Fast categorical encoding of userId and movieId...")

    # Load filtered ratings
    df = pl.read_csv(FILTERED_RATINGS_FILE)

    # Convert to categorical and get codes (extremely fast)
    df_encoded = df.with_columns(
        [
            pl.col("userId").cast(pl.Categorical).to_physical().alias("user_index"),
            pl.col("movieId").cast(pl.Categorical).to_physical().alias("movie_index"),
        ]
    )

    # Create mappings from the categorical data
    user_cats = df.select("userId").unique().sort("userId").cast(pl.Categorical)
    movie_cats = df.select("movieId").unique().sort("movieId").cast(pl.Categorical)

    user2index = {
        user: idx
        for idx, user in enumerate(user_cats["userId"].cast(pl.String).to_list())
    }

    movie2index = {
        movie: idx
        for idx, movie in enumerate(movie_cats["movieId"].cast(pl.String).to_list())
    }

    # Save results
    os.makedirs(ENCODED_DIR, exist_ok=True)
    df_encoded.write_csv(ENCODED_RATINGS_FILE)

    with open(USER2INDEX_FILE, "w") as f:
        json.dump(user2index, f)

    with open(MOVIE2INDEX_FILE, "w") as f:
        json.dump(movie2index, f)

    print(f" Encoded ratings → {ENCODED_RATINGS_FILE}")
    print(f" user2index → {USER2INDEX_FILE} ({len(user2index)} users)")
    print(f" movie2index → {MOVIE2INDEX_FILE} ({len(movie2index)} movies)")

    return df_encoded


def encode_ids_lazy():
    """
    Memory-efficient version using lazy evaluation.
    Best for datasets that don't fit in memory.
    """
    print(" [Polars] Memory-efficient lazy encoding...")

    # Use lazy frame for memory efficiency
    df_lazy = pl.scan_csv(FILTERED_RATINGS_FILE)

    # Create mappings using lazy operations
    user_mapping = (
        df_lazy.select("userId")
        .unique()
        .sort("userId")
        .with_columns(pl.int_range(pl.len()).alias("user_index"))
        .collect()  # Only collect the small mapping
    )

    movie_mapping = (
        df_lazy.select("movieId")
        .unique()
        .sort("movieId")
        .with_columns(pl.int_range(pl.len()).alias("movie_index"))
        .collect()  # Only collect the small mapping
    )

    # Process and save in streaming fashion
    df_encoded = df_lazy.join(user_mapping.lazy(), on="userId", how="left").join(
        movie_mapping.lazy(), on="movieId", how="left"
    )

    # Stream to CSV without loading full dataset in memory
    os.makedirs(ENCODED_DIR, exist_ok=True)
    df_encoded.sink_csv(ENCODED_RATINGS_FILE)

    # Save mappings
    user2index = dict(
        zip(user_mapping["userId"].to_list(), user_mapping["user_index"].to_list())
    )

    movie2index = dict(
        zip(movie_mapping["movieId"].to_list(), movie_mapping["movie_index"].to_list())
    )

    with open(USER2INDEX_FILE, "w") as f:
        json.dump(user2index, f)

    with open(MOVIE2INDEX_FILE, "w") as f:
        json.dump(movie2index, f)

    print(f" Encoded ratings → {ENCODED_RATINGS_FILE}")
    print(f" user2index → {USER2INDEX_FILE} ({len(user2index)} users)")
    print(f" movie2index → {MOVIE2INDEX_FILE} ({len(movie2index)} movies)")


if __name__ == "__main__":
    print(" Starting encoding process...")
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Choose which encoding method to use
        encode_ids()  # Use the standard version
        # encode_ids_categorical()  # Or use the categorical version
        # encode_ids_lazy()  # Or use the lazy version for large datasets

        print(" Encoding completed successfully!")

    except Exception as e:
        print(f" Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
