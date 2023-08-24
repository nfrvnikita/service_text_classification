# pylint: disable=invalid-name
"""Импорт библиотек"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def text_vectorize(X_train=np.array, X_valid=np.array):
    """Токенизируем отзывы при помощи CountVectorizer,
    затем мы создаём TF-IDF версии токенизированных твитов.

    Args:
        X_train (np.array): тренировочный набор отзывов.
        X_valid (np.array): валидационный набор отзывов.
    """
    clf = CountVectorizer()
    X_train_cv = clf.fit_transform(X_train)
    X_test_cv = clf.transform(X_valid)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)

    return X_train_tf, X_test_tf
