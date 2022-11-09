# -*- coding:utf-8 -*-
"""
@file: p_top_k_evaluation_script.py
@time: 31/10/2022 17:41
@desc: script to evaluate the precision@k of different methods.
@author: Echo
"""
from ext import preprocessing
from recommender_system.content_based import ContentBasedRecommender
from recommender_system.evaluation import Evaluation


def evaluation_p_top_k(user_id: int, test_size: float, k: int, retrieval_method=None):
    """
    evaluate the precision@k of different methods.
    :param user_id: user id.
    :param test_size: test size
    :param k: number of top k.
    :param retrieval_method: there are choices between faiss, lsh and basic content based algorithm,
                            if it is None, then it is basic content based algorithm.
    :return: p_top_k: the value of precision@k.
    """
    _, true_test_movies = preprocessing.dataset_split(user_id, test_size)

    """content based recommendation"""
    content_recommender = ContentBasedRecommender()
    predict_movies_ids = list(
        content_recommender.get_ordered_rankings(user_id, test_size, retrieval_method=retrieval_method, top_k=k).keys())

    true_test_movies = list(true_test_movies)
    true_test_movies_ratings = preprocessing.get_one_user_ratings_with_movies_ids(user_id, true_test_movies)

    true_test_movies_ids = []
    for item in true_test_movies_ratings.items():
        if item[1] >= 3:
            true_test_movies_ids.append(item[0])
    p_top_k = Evaluation().precision_top_k(true_test_movies_ids, predict_movies_ids[:k], k)
    return p_top_k

