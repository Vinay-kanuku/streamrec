import polars as pl
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from src.config import (
    MATRIX_DIR,
    ENCODED_RATINGS_FILE,
    USER_ITEM_MATRIX_FILE
)

import os 
def build_user_item_matrix():
    """Optimized version using Polars native operations and memory-efficient processing."""
    print("🧱 Building user-item sparse matrix...")
    
    # Read only required columns to save memory
    df = pl.read_csv(ENCODED_RATINGS_FILE, columns=["user_index", "movie_index", "rating"])
    
    # Get matrix dimensions efficiently using Polars
    stats = df.select([
        pl.col("user_index").max().alias("max_user"),
        pl.col("movie_index").max().alias("max_movie"),
        pl.len().alias("nnz")
    ]).row(0)
    
    num_users = stats[0] + 1
    num_movies = stats[1] + 1
    nnz = stats[2]
    
    print(f"📐 Matrix dimensions: {num_users} users × {num_movies} movies")
    print(f"📊 Non-zero entries: {nnz:,}")
    
    # Convert to numpy arrays in one go (more efficient than individual conversions)
    arrays = df.select([
        pl.col("user_index"),
        pl.col("movie_index"), 
        pl.col("rating")
    ]).to_numpy()
    
    user_indices = arrays[:, 0].astype(np.int32)  # Use int32 to save memory
    movie_indices = arrays[:, 1].astype(np.int32)
    ratings = arrays[:, 2].astype(np.float32)
    
    # Build sparse matrix
    matrix = csr_matrix(
        (ratings, (user_indices, movie_indices)),
        shape=(num_users, num_movies),
        dtype=np.float32
    )
    
    # Eliminate zeros and duplicates for better compression
    matrix.eliminate_zeros()
    matrix.sum_duplicates()
    
    # Save with compression
    os.makedirs(MATRIX_DIR)
    user_item_matrix = os.path.join(MATRIX_DIR, USER_ITEM_MATRIX_FILE)
    save_npz(user_item_matrix, matrix, compressed=True)
    
    # Calculate density
    density = matrix.nnz / (num_users * num_movies)
    
    print(f"✅ Sparse matrix shape: {matrix.shape}")
    print(f"📊 Density: {density:.6f} ({density*100:.4f}%)")
    print(f"💾 Saved to: {USER_ITEM_MATRIX_FILE}")
    print(f"🗜️  Compression ratio: {nnz/matrix.nnz:.2f}x" if matrix.nnz != nnz else "🗜️  No duplicates found")
    
    return matrix


def build_user_item_matrix_lazy():
    """Memory-efficient version for very large datasets using lazy evaluation."""
    print("🧱 Building user-item sparse matrix (lazy mode)...")
    
    # Use lazy frame for memory efficiency
    df_lazy = pl.scan_csv(ENCODED_RATINGS_FILE)
    
    # Get dimensions first
    stats = (
        df_lazy
        .select([
            pl.col("user_index").max().alias("max_user"),
            pl.col("movie_index").max().alias("max_movie"),
            pl.len().alias("nnz")
        ])
        .collect()
        .row(0)
    )
    
    num_users = stats[0] + 1
    num_movies = stats[1] + 1
    nnz = stats[2]
    
    print(f"📐 Matrix dimensions: {num_users} users × {num_movies} movies")
    print(f"📊 Non-zero entries: {nnz:,}")
    
    # Process data in streaming fashion
    arrays = (
        df_lazy
        .select([
            pl.col("user_index").cast(pl.Int32),
            pl.col("movie_index").cast(pl.Int32),
            pl.col("rating").cast(pl.Float32)
        ])
        .collect()
        .to_numpy()
    )
    
    user_indices = arrays[:, 0]
    movie_indices = arrays[:, 1] 
    ratings = arrays[:, 2]
    
    # Build and optimize matrix
    matrix = csr_matrix(
        (ratings, (user_indices, movie_indices)),
        shape=(num_users, num_movies),
        dtype=np.float32
    )
    
    matrix.eliminate_zeros()
    matrix.sum_duplicates()
    
    # Save with compression
    save_npz(USER_ITEM_MATRIX_FILE, matrix, compressed=True)
    
    density = matrix.nnz / (num_users * num_movies)
    
    print(f"✅ Sparse matrix shape: {matrix.shape}")
    print(f"📊 Density: {density:.6f} ({density*100:.4f}%)")
    print(f"💾 Saved to: {USER_ITEM_MATRIX_FILE}")
    
    return matrix


def build_user_item_matrix_chunked(chunk_size=100000):
    """
    Process very large datasets in chunks to minimize memory usage.
    Best for datasets > 10M ratings.
    """
    print(f"🧱 Building user-item sparse matrix (chunked: {chunk_size:,} rows)...")
    
    # First pass: get dimensions
    df_stats = pl.read_csv(ENCODED_RATINGS_FILE, columns=["user_index", "movie_index"])
    stats = df_stats.select([
        pl.col("user_index").max().alias("max_user"),
        pl.col("movie_index").max().alias("max_movie")
    ]).row(0)
    
    num_users = stats[0] + 1
    num_movies = stats[1] + 1
    
    print(f"📐 Matrix dimensions: {num_users} users × {num_movies} movies")
    
    # Initialize lists for matrix construction
    all_user_indices = []
    all_movie_indices = []
    all_ratings = []
    
    # Process in chunks
    df_lazy = pl.scan_csv(ENCODED_RATINGS_FILE)
    total_rows = df_lazy.select(pl.len()).collect().item()
    
    for i in range(0, total_rows, chunk_size):
        print(f"Processing chunk {i//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size}...")
        
        chunk = (
            df_lazy
            .slice(i, chunk_size)
            .select([
                pl.col("user_index").cast(pl.Int32),
                pl.col("movie_index").cast(pl.Int32),
                pl.col("rating").cast(pl.Float32)
            ])
            .collect()
        )
        
        arrays = chunk.to_numpy()
        all_user_indices.append(arrays[:, 0])
        all_movie_indices.append(arrays[:, 1])
        all_ratings.append(arrays[:, 2])
    
    # Combine all chunks
    user_indices = np.concatenate(all_user_indices)
    movie_indices = np.concatenate(all_movie_indices)
    ratings = np.concatenate(all_ratings)
    
    # Build matrix
    matrix = csr_matrix(
        (ratings, (user_indices, movie_indices)),
        shape=(num_users, num_movies),
        dtype=np.float32
    )
    
    matrix.eliminate_zeros()
    matrix.sum_duplicates()
    
    # Save with compression
    save_npz(USER_ITEM_MATRIX_FILE, matrix, compressed=True)
    
    density = matrix.nnz / (num_users * num_movies)
    
    print(f"✅ Sparse matrix shape: {matrix.shape}")
    print(f"📊 Density: {density:.6f} ({density*100:.4f}%)")
    print(f"💾 Saved to: {USER_ITEM_MATRIX_FILE}")
    
    return matrix


def validate_matrix(matrix_file=USER_ITEM_MATRIX_FILE):
    """Quick validation of the saved matrix."""
    from scipy.sparse import load_npz
    
    matrix = load_npz(matrix_file)
    print(f"🔍 Matrix validation:")
    print(f"   Shape: {matrix.shape}")
    print(f"   Non-zeros: {matrix.nnz:,}")
    print(f"   Data type: {matrix.dtype}")
    print(f"   Memory usage: {matrix.data.nbytes / 1024**2:.2f} MB")
    
    return matrix