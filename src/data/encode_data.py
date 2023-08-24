"""Импорт библиотек"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def ohe_encoding(train_y=np.array, valid_y=np.array):
    """_summary_

    Args:
        train_y (_type_, optional): _description_. Defaults to np.array.
        valid_y (_type_, optional): _description_. Defaults to np.array.
    """
    oh_encoder = OneHotEncoder()
    train_y = oh_encoder.fit_transform(np.array(train_y).reshape(-1, 1)).toarray()
    valid_y = oh_encoder.fit_transform(np.array(valid_y).reshape(-1, 1)).toarray()
    return train_y, valid_y
