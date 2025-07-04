from src.models.collaborative_filterring import recommend_items 
from src.config import ENCODED_RATINGS_FILE, MOVIE2INDEX_FILE
import polars as pl 
import json 
js = json.loads(str(MOVIE2INDEX_FILE))

def get_movie_title(df: pl.DataFrame, movie_id: int) -> str | None:
    result = df.filter(pl.col("movieId") == movie_id)
    return result["title"][0] if result.height > 0 else None


user_id = int(input("Enter the user ID: "))
recommendations, scores = recommend_items(user_id, top_k=5)
print(f"Recommendations for user {user_id}: {recommendations}")
print(f"Scores: {scores}")
df = pl.read_csv(ENCODED_RATINGS_FILE)
for rec in recommendations:
    print(get_movie_title(df, js[rec]))

