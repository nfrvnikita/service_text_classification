"""Импорт библиотек"""
import pandas as pd
from transformers import BertTokenizer


TOKENIZER_PATH = 'cointegrated/rubert-tiny'


def create_lens(data=pd.DataFrame) -> pd.DataFrame:
    """Генерируем признак с кол-вом слов в отзыве.

    Args:
        data (pd.DataFrame): датасет

    Returns:
        pd.DataFrame: датасет с кол-вом слов.
    """
    text_len = []
    for text in data['clean_review']:
        tweet_len = len(text.split())
        text_len.append(tweet_len)
    data['text_len'] = text_len

    return data[data['text_len'] > 4]


def create_tokens(data=pd.DataFrame, path=TOKENIZER_PATH) -> pd.DataFrame:
    """Токенизируем текст.

    Args:
        data (pd.DataFrame): датасет.
        path: путь до BERT токенизатора.

    Returns:
        pd.DataFrame: датасет с токенами.
    """
    tokenizer = BertTokenizer.from_pretrained(path)

    token_lens = []
    for txt in data['clean_review'].values:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens.append(len(tokens))
    data['token_lens'] = token_lens
    data = data.sort_values(by='token_lens', ascending=False)
    data = data.sample(frac=1).reset_index(drop=True)

    return data


def classify_rating(value=str):
    """Генерируем таргет для мультиклассовой классификации.

    Args:
        value (str): значение таргета.
    """
    # Вернем None для значений, выходящих за пределы диапазона
    if value < 0 or value > 100:
        return None
    # Границы диапазонов
    bins = [0, 40, 70, 100]
    # Классы
    labels = [0, 1, 2]
    return pd.cut([value], bins=bins, labels=labels)[0]
