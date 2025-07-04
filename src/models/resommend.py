from abc import ABC, abstractmethod 
from scipy.sparse import load_npz 
from src.config import USER_ITEM_MATRIX_FILE 
from sklearn.preprocessing import normalize  
from sklearn.metrics.pairwise import cosine_similarity 
from surprise import Reader, SVD , Dataset
from surprise.model_selection import (GridSearchCV, train_test_split)
import pandas as pd 


def load_matrix():
    R = load_npz(USER_ITEM_MATRIX_FILE)
    return R 

class Recommender(ABC):
    @abstractmethod
    def recommend(self):
        pass 

class ItemBasedFiltering(Recommender):
    def __init__(self): 
        pass 

    def recommend(self, user_id):
        try:
            R = load_matrix()
            R_normalised = normalize(R, axis=0)
            pass 
        except Exception as e:
            raise(e) 
        
    









class UserBasedFiltering(Recommender):
    def __init__(self):
        super().__init__()

    def recommend(self):
        return super().recommend()
    
class MatrixFactorisationFiltering(Recommender):
    def __init__(self):
        super().__init__()

    def recommend(self):
        return super().recommend()
    

class MovieRecommender:
    def __init__(self, recomender:Recommender):
        self.recommender = recomender

    def recommed_movie(self, user_id:int):
        movies = self.recommender.recommend(user_id)
        return movies 
    




if __name__ == "__main__":
    item = ItemBasedFiltering()
    ob = MovieRecommender(item)
    ob.recommed_movie(30)
