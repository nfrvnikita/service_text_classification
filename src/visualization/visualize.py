"""Импорт библиотек"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def plot_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    metric_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metric_dict.keys()), y=list(metric_dict.values()))
    plt.title('Model Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    st.pyplot(plt)
