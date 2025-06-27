from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BUILD_DIR = BASE_DIR / "processed_data"

# Raw data files
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"

# Clean data
CLEAN_DIR = BUILD_DIR / "clean"
CLEAN_RATINGS_FILE = CLEAN_DIR / "ratings_clean.csv"  # FIXED: Full path
CLEAN_MOVIES_FILE = CLEAN_DIR / "movies_clean.csv"    # FIXED: Full path

# Filtered data
FILTER_DIR = BUILD_DIR / "filtered"
FILTERED_RATINGS_FILE = FILTER_DIR / "ratings_filtered.csv"

# Encoded data
ENCODED_DIR = BUILD_DIR / "encoded"
ENCODED_RATINGS_FILE = ENCODED_DIR / "ratings_encoded.csv"
USER2INDEX_FILE = ENCODED_DIR / "user2index.json"
MOVIE2INDEX_FILE = ENCODED_DIR / "movie2index.json"

# Matrix data
MATRIX_DIR = BUILD_DIR / "matrix"
USER_ITEM_MATRIX_FILE = MATRIX_DIR / "user_item_matrix.npz"

# Content data
CONTENT_DIR = BUILD_DIR / "content"
GENRE_TFIDF_FILE = CONTENT_DIR / "genre_tfidf.npy"
MOVIEID2ROW_FILE = CONTENT_DIR / "movieid2row.json"

# Parameters
MIN_USER_RATINGS = 10
MIN_MOVIE_RATINGS = 10
USE_TFIDF = True
MAX_FEATURES = 50