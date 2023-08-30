"""Импортируем библиотеки"""
import re
import pandas as pd
from src.features.build_features import (create_lens,
                                         create_tokens,
                                         classify_rating)


def text_preprocessing(text=str) -> str:
    """Чистим текст от символов, понижаем регистр.

    Args:
        text (str): столбец, который будет обработан.

    Returns:
        str: обработанная строка.
    """
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    regular = r'[\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+\:\\+\«+\»+\—]'
    regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)'
    text = re.sub(regular, '', text)
    text = re.sub(regular_url, r'URL', text)
    text = text.replace('\n', ' ')
    text = text.replace('%', ' <проценты>')
    return text


def cleaning(data=pd.DataFrame, column='review') -> pd.DataFrame:
    """Применяем все функции по очистке текста.

    Args:
        data (pd.DataFrame): сырой датасет.
        column (pd.Series): столбец, который будем обрабатывать.

    Returns:
        pd.DataFrame: обработанный датасет.
    """
    data = data.drop(columns=['published', 'full review url'])
    texts_new = []
    for text in data[column]:
        texts_new.append(text_preprocessing(text))
    data['clean_review'] = texts_new
    return data


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
