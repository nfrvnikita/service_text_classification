"""Импортируем библиотеки"""
import re
import pandas as pd


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


def cleaning(data=pd.DataFrame, column=pd.Series) -> pd.DataFrame:
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
