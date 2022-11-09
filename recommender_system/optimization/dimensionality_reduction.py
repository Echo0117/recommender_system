# -*- coding:utf-8 -*-
"""
@file: dimensionality_reduction.py
@time: 09/10/2022 20:29
@desc: optimization using svd
@author: Echo
"""
from surprise import SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


class DimensionReduce():
    def __init__(self, data):
        self.reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp')
        self.data = Dataset.load_from_df(data, self.reader)
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2)

    def svd(self):
        algo = SVD()
        algo.fit(self.trainset)
        return algo

    def predict_rmse(self, algo):
        predictions = algo.test(self.testset)
        rmse = accuracy.rmse(predictions)
        return rmse
