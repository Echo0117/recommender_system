# -*- coding:utf-8 -*-
"""
@file: test_evaluation.py
@time: 25/10/2022 08:49
@desc: unit tests for evaluations class
@author: Echo
"""
import pytest
from recommender_system.evaluation import Evaluation


class TestEvaluation(object):
    def test_cosine_similarity(self):
        matrix_a = [[1, 0, -1]]
        matrix_b = [[-1, -1, 0], [-1, -1, 0]]
        cos_sim = Evaluation().get_cosine_similarity(matrix_a, matrix_b)
        assert list(cos_sim) == [-0.4999999999999999, -0.4999999999999999]

    def test_precision_top_k(self):
        matrix_a = [1, 0, -1, -1, -1, 0]
        matrix_b = [-1, -1, 0, -1, -1, 0]
        p_top_k = Evaluation().precision_top_k(matrix_a, matrix_b, 6)
        assert p_top_k == 0.3333333333333333


if __name__ == "__main__":
    pytest.main()
