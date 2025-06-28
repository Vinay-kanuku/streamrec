from preprocessing.build_matrix import build_user_item_matrix
from src.models.collaborative_filterring import recommend_items
from src.preprocessing.cleaning import clean_movielens_data
from src.preprocessing.encoding import encode_ids
from src.preprocessing.filter_sparse import filter_sparse_users_movies

clean_movielens_data()
filter_sparse_users_movies()
print(encode_ids())
print(build_user_item_matrix())
user_id = 20
recommendations, scores = recommend_items(user_id, top_k=5)
print(f"Recommendations for user {user_id}: {recommendations}")
print(f"Scores: {scores}")
