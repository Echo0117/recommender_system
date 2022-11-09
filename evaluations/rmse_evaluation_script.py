# -*- coding:utf-8 -*-
"""
@file: rmse_evaluation_script.py
@time: 31/10/2022 16:56
@desc: script to evaluate the rmse of different methods
@author: Echo
"""
from ext import preprocessing
from recommender_system.collaborative_filtering import CollaborativeFilteringUsersBased
from recommender_system.evaluation import Evaluation


def evaluation_with_rmse(user_id: int, test_size: float):
    """
    evaluate the rmse of different methods.
    :param user_id: user id
    :param test_size: test size
    :return: rmse: the value of rmse.
    """
    _, true_test_movies = preprocessing.dataset_split(user_id, test_size)

    """CollaborativeFilteringUsersBased recommendation"""
    cf_recommender = CollaborativeFilteringUsersBased()
    predict_movies = cf_recommender.predicting_ordered_ratings(user_id, test_size)

    true_test_movies = list(true_test_movies)
    true_test_movies_ratings = preprocessing.get_one_user_ratings_with_movies_ids(user_id, true_test_movies)

    sorted_predict_movies = sorted(predict_movies.items(), key=lambda x: x[0])
    sorted_true_test_movies = sorted(true_test_movies_ratings.items(), key=lambda x: x[0])

    sorted_predict_movies = [item[1] for item in sorted_predict_movies]
    sorted_true_test_movies = [item[1] for item in sorted_true_test_movies]

    rmse = Evaluation().root_mean_square_error(sorted_true_test_movies, sorted_predict_movies)
    return rmse
