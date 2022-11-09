# -*- coding:utf-8 -*-
"""
@file: evaluation.py
@time: 09/10/2022 20:24
@desc: 
@author: Echo
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


class Evaluation:
    def __int__(self):
        pass

    def root_mean_square_error(self, y_true, y_pred) -> float:
        """
        evaluation with mean square error
        :param y_true:
        :param y_pred:
        :return:
        """
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        mse = mean_squared_error(y_pred, y_true)
        return sqrt(mse)

    def precision_top_k(self, y_true, y_pred, k) -> float:
        """
        Precision @ k = (  # of recommended items @k that are relevant) / (# of recommended items @k)

        :param y_true:
        :param y_pred:
        :return:
        """
        """we calculate the relevant list by getting the intersection set form the recommended items and true items 
        that the user watched and liked """
        true_positives_list = list(set(y_pred).intersection(set(y_true)))
        p_top_k = len(true_positives_list) / k
        print(true_positives_list)
        return p_top_k

    def get_cosine_similarity(self, matrix_a, matrix_b) -> np.ndarray:
        """
        get the cosine similarity between matrix_a and matrix_b
        :param matrix_a: 2d matrix
        :param matrix_b: 2d matrix
        :return: cosine similarity
        """
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)
        cos_sim = np.sum(matrix_a * matrix_b, axis=1) / (
                np.linalg.norm(matrix_a, axis=1) * np.linalg.norm(matrix_b, axis=1))
        return cos_sim
