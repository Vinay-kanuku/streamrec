import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import os
from typing import Tuple, Optional
from src.config import USER_ITEM_MATRIX_FILE

class OptimizedCollaborativeFilter:
    def __init__(self, similarity_cache_file: str = "item_similarity_cache.pkl"):
        self.matrix = None
        self.item_similarity = None
        self.similarity_cache_file = similarity_cache_file
        self.user_means = None
        
    def load_matrix(self):
        """Load and prep are the user-item matrix."""
        if self.matrix is None:
            print("📥 Loading user-item matrix...")
            self.matrix = load_npz(USER_ITEM_MATRIX_FILE).tocsr()
            print(f"✅ Matrix loaded: {self.matrix.shape}")
            
            # Precompute user means for mean-centered recommendations
            self.user_means = np.array(self.matrix.mean(axis=1)).flatten()
            
    def _compute_item_similarity(self, use_cache: bool = True, min_interactions: int = 5) -> np.ndarray:
        """
        Compute item-item similarity matrix with optimizations.
        
        Args:
            use_cache: Whether to use/save cached similarity matrix
            min_interactions: Minimum interactions required for an item to be considered
        """
        if use_cache and os.path.exists(self.similarity_cache_file):
            print("📁 Loading cached item similarity matrix...")
            with open(self.similarity_cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("🔄 Computing item-item similarity matrix...")
        
        # Filter items with minimum interactions to reduce noise
        item_counts = np.array(self.matrix.sum(axis=0)).flatten()
        valid_items = item_counts >= min_interactions
        
        if valid_items.sum() < self.matrix.shape[1]:
            print(f"🔍 Filtering items: {valid_items.sum()}/{self.matrix.shape[1]} items have ≥{min_interactions} interactions")
            filtered_matrix = self.matrix[:, valid_items]
        else:
            filtered_matrix = self.matrix
            
        # Use TF-IDF weighting to reduce popularity bias
        tfidf = TfidfTransformer()
        weighted_matrix = tfidf.fit_transform(filtered_matrix.T).T
        
        # Compute cosine similarity with better memory efficiency
        if filtered_matrix.shape[1] > 10000:
            # For large matrices, compute in chunks
            similarity = self._compute_similarity_chunked(weighted_matrix.T)
        else:
            similarity = cosine_similarity(weighted_matrix.T, dense_output=False)
            if hasattr(similarity, 'toarray'):
                similarity = similarity.toarray()
        
        # Expand back to full size if we filtered
        if valid_items.sum() < self.matrix.shape[1]:
            full_similarity = np.zeros((self.matrix.shape[1], self.matrix.shape[1]))
            valid_indices = np.where(valid_items)[0]
            full_similarity[np.ix_(valid_indices, valid_indices)] = similarity
            similarity = full_similarity
        
        # Cache the similarity matrix
        if use_cache:
            print("💾 Caching similarity matrix...")
            with open(self.similarity_cache_file, 'wb') as f:
                pickle.dump(similarity, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        return similarity
    
    def _compute_similarity_chunked(self, matrix: csr_matrix, chunk_size: int = 1000) -> np.ndarray:
        """Compute cosine similarity in chunks to save memory."""
        n_items = matrix.shape[0]
        similarity = np.zeros((n_items, n_items), dtype=np.float32)
        
        for i in range(0, n_items, chunk_size):
            end_i = min(i + chunk_size, n_items)
            chunk_i = matrix[i:end_i]
            
            for j in range(i, n_items, chunk_size):
                end_j = min(j + chunk_size, n_items)
                chunk_j = matrix[j:end_j]
                
                # Compute similarity for this chunk
                sim_chunk = cosine_similarity(chunk_i, chunk_j, dense_output=True)
                similarity[i:end_i, j:end_j] = sim_chunk
                
                # Fill symmetric part (except diagonal)
                if i != j:
                    similarity[j:end_j, i:end_i] = sim_chunk.T
                    
            print(f"   Progress: {min(end_i, n_items)}/{n_items} items processed")
        
        return similarity
    
    def recommend_items_optimized(self, user_index: int, top_k: int = 10, 
                                 use_mean_centering: bool = True,
                                 similarity_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized item recommendation with multiple improvements.
        
        Args:
            user_index: Index of the user to recommend for
            top_k: Number of recommendations to return
            use_mean_centering: Whether to use mean-centered ratings
            similarity_threshold: Minimum similarity to consider
        """
        self.load_matrix()
        
        if user_index >= self.matrix.shape[0]:
            raise ValueError(f"User index {user_index} out of range (max: {self.matrix.shape[0]-1})")
        
        print(f"🤖 Generating recommendations for user {user_index}...")
        
        # Get user's ratings
        user_ratings = self.matrix.getrow(user_index)
        
        if user_ratings.nnz == 0:
            print("⚠️  User has no ratings, returning popular items...")
            return self._recommend_popular_items(top_k)
        
        # Load/compute item similarity
        if self.item_similarity is None:
            self.item_similarity = self._compute_item_similarity()
        
        print("⚙️  Computing predicted ratings...")
        
        # Get only items the user has rated for efficiency
        rated_items = user_ratings.indices
        user_ratings_dense = user_ratings.toarray().flatten()
        
        if use_mean_centering:
            # Mean-center the ratings
            user_mean = self.user_means[user_index]
            centered_ratings = user_ratings_dense[rated_items] - user_mean
        else:
            centered_ratings = user_ratings_dense[rated_items]
            user_mean = 0
        
        # Compute scores only for unrated items (much faster)
        unrated_items = np.setdiff1d(np.arange(self.matrix.shape[1]), rated_items)
        
        if len(unrated_items) == 0:
            print("⚠️  User has rated all items!")
            return np.array([]), np.array([])
        
        # Vectorized computation for unrated items only
        similarity_subset = self.item_similarity[np.ix_(unrated_items, rated_items)]
        
        # Apply similarity threshold
        similarity_subset[similarity_subset < similarity_threshold] = 0
        
        # Compute predictions
        numerator = similarity_subset.dot(centered_ratings)
        denominator = np.abs(similarity_subset).sum(axis=1)
        
        # Avoid division by zero
        valid_predictions = denominator > 0
        scores = np.full(len(unrated_items), -np.inf)
        scores[valid_predictions] = (numerator[valid_predictions] / 
                                   denominator[valid_predictions]) + user_mean
        
        # Get top-K recommendations
        if len(scores) < top_k:
            top_k = len(scores)
            
        top_indices_local = np.argpartition(scores, -top_k)[-top_k:]
        top_indices_local = top_indices_local[np.argsort(-scores[top_indices_local])]
        
        # Map back to global item indices
        top_indices = unrated_items[top_indices_local]
        top_scores = scores[top_indices_local]
        
        print(f"✅ Generated {len(top_indices)} recommendations")
        return top_indices, top_scores
    
    def _recommend_popular_items(self, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to popular items for cold-start users."""
        item_popularity = np.array(self.matrix.sum(axis=0)).flatten()
        top_indices = np.argpartition(item_popularity, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-item_popularity[top_indices])]
        return top_indices, item_popularity[top_indices]
    
    def batch_recommend(self, user_indices: list, top_k: int = 10) -> dict:
        """Generate recommendations for multiple users efficiently."""
        self.load_matrix()
        
        if self.item_similarity is None:
            self.item_similarity = self._compute_item_similarity()
        
        recommendations = {}
        
        print(f"🎯 Generating recommendations for {len(user_indices)} users...")
        
        for i, user_idx in enumerate(user_indices):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(user_indices)} users processed")
                
            try:
                items, scores = self.recommend_items_optimized(user_idx, top_k)
                recommendations[user_idx] = {
                    'items': items.tolist(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                print(f"   Error for user {user_idx}: {e}")
                recommendations[user_idx] = {'items': [], 'scores': []}
        
        return recommendations


# Convenience functions for backward compatibility
_recommender = OptimizedCollaborativeFilter()

def recommend_items(user_index: int, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized recommendation function with caching and multiple improvements.
    
    Performance improvements:
    - Cached similarity matrix computation
    - TF-IDF weighting to reduce popularity bias  
    - Mean-centered ratings for better predictions
    - Similarity thresholding to reduce noise
    - Chunked computation for large matrices
    - Only compute predictions for unrated items
    """
    return _recommender.recommend_items_optimized(user_index, top_k)

def batch_recommend_users(user_indices: list, top_k: int = 10) -> dict:
    """Generate recommendations for multiple users."""
    return _recommender.batch_recommend(user_indices, top_k)

def clear_cache():
    """Clear the similarity matrix cache."""
    if os.path.exists(_recommender.similarity_cache_file):
        os.remove(_recommender.similarity_cache_file)
        _recommender.item_similarity = None
        print("🗑️  Cache cleared")


# Example usage and testing
if __name__ == "__main__":
    # Test single user recommendation
    user_id = 0
    recommendations, scores = recommend_items(user_id, top_k=5)
    print(f"Recommendations for user {user_id}: {recommendations}")
    print(f"Scores: {scores}")
    
    # Test batch recommendations
    # batch_recs = batch_recommend_users([0, 1, 2], top_k=5)
    # print(f"Batch recommendations: {batch_recs}")