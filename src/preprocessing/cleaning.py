import pandas as pd
from src.config import RATINGS_FILE, MOVIES_FILE, CLEAN_DIR, CLEAN_RATINGS_FILE, CLEAN_MOVIES_FILE
import os

def clean_movielens_data():
    """Clean MovieLens data and save to clean directory"""
    
  
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    print(len(ratings))
    print(len(movies))



    ratings.drop_duplicates()
    movies.drop_duplicates()

    print(len(ratings))
    print(len(movies)) 

    
    os.makedirs(CLEAN_DIR, exist_ok=True)
    clean_ratings_path = os.path.join(CLEAN_DIR, CLEAN_RATINGS_FILE)
    clean_movies_path = os.path.join(CLEAN_DIR, CLEAN_MOVIES_FILE)
    
    ratings.to_csv(clean_ratings_path, index=False)
    movies.to_csv(clean_movies_path, index=False)
    
    return ratings, movies

if __name__ == "__main__":
    clean_movielens_data()