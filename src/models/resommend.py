import multiprocessing
from abc import ABC, abstractmethod

import numpy as np


import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.config import USER_ITEM_MATRIX_FILE


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
            temp = R_normalised[:10000, :10000]
            temp = temp.T
            rows = temp.shape[0]
            block_size = 20
            ans = []
            tasks = [(i, min(i + block_size, rows), temp) for i in range(rows)]
            with multiprocessing.Pool(processes=8) as pro:
                res = pro.starmap(self.compute_block, tasks)

            ans = np.vstack(res)
            print(ans)

        except Exception as e:
            raise (e)

    def compute_block(self, start, stop, X):
        return cosine_similarity(X[start:stop], X)

    def __compute_similarity(self, user_tem_matrix):
        """
        Computes the similarity of the user item matrxi
        args:
        Returns: similari matrxi
        """


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
    def __init__(self, recomender: Recommender):
        self.recommender = recomender

    def recommed_movie(self, user_id: int):
        movies = self.recommender.recommend(user_id)
        return movies


if __name__ == "__main__":
    item = ItemBasedFiltering()
    ob = MovieRecommender(item)
    ob.recommed_movie(30)
