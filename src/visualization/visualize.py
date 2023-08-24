"""Импорт библиотек"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def count_words_review(data=pd.DataFrame):
    """Функция дли визуализации количества слов < 10

    Args:
        data (pd.DataFrame): датасет с данными.
    """
    plt.figure(figsize=(7, 5))
    axis = sns.countplot(x='text_len', data=data[data['text_len'] < 10], palette='mako')
    plt.title('Отзывы, содержащие меньше 10 слов')
    plt.yticks([])
    axis.bar_label(axis.containers[0])
    plt.ylabel('Кол-во')
    plt.xlabel('')
    plt.show()
