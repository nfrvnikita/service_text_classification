# pylint: disable=invalid-name
"""Импорт библиотек"""
import pandas as pd
from sklearn.model_selection import train_test_split


def split(data_os=pd.DataFrame, seed=42):
    """_summary_

    Args:
        data_os (_type_, optional): _description_. Defaults to pd.DataFrame.
    """
    X = data_os['clean_review'].values
    y = data_os['class'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.25,
                                                          stratify=y,
                                                          random_state=seed)
    return X_train, X_valid, y_train, y_valid
