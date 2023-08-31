# pylint: disable=invalid-name
"""Импорт библиотек"""
import pandas as pd


PATH_TO_SAVE = '/home/nfrvnikita/projects/service4classification/data/interim/'


def split_dataset(data=pd.DataFrame, train_ratio=0.85):
    """_summary_

    Args:
        data (_type_, optional): _description_. Defaults to pd.DataFrame.
        train_ratio (float, optional): _description_. Defaults to 0.85.
    """
    train_size = int(len(data) * train_ratio)
    train_set = data[:train_size]
    test_set = data[train_size:]

    train_set.to_csv(f'{PATH_TO_SAVE}train_set.csv', index=False)
    test_set.to_csv(f'{PATH_TO_SAVE}test_set.csv', index=False)
