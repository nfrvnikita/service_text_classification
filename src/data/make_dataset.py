"""Импорт библиотек"""
import pandas as pd
from src.data.preprocessing import cleaning
from src.features.build_features import (create_lens,
                                         create_tokens,
                                         classify_rating)


def final_dataset(data=pd.DataFrame) -> pd.DataFrame:
    """Применяем функции для генерации финального датасета.

    Args:
        data (pd.DataFrame):

    Returns:
        pd.DataFrame: готовый датасет
    """
    data = cleaning(data, 'review')
    data = create_tokens(create_lens(data))
    data['class'] = data['rating'].apply(classify_rating)
    return data
