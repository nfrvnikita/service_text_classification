# pylint: disable=invalid-name
"""Импорт библиотек"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def sampling_balance(data=pd.DataFrame) -> pd.DataFrame:
    """Функция для решения проблемы с дисбалансом классов.

    Args:
        data (pd.DataFrame): финальный датасет.

    Returns:
        pd.DataFrame: сэмплированный датасет.
    """
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(np.array(data['clean_review']).reshape(-1, 1),
                                        np.array(data['class']).reshape(-1, 1))
    return pd.DataFrame(list(zip([x[0] for x in X_train], y_train)),
                        columns=['clean_review', 'class'])
