"""Импорт библиотек"""
# import pandas as pd
# import torch
from src.data.preprocessing import final_dataset
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    """Кастомизируем Dataset для подготовки данных к обучению"""
    def __init__(self, data, tokenizer):
        self.data = final_dataset(data)
        self.targets = list(self.data['class'])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = self.data['review'][index]
        target = self.targets[index]

        encoded_review = self.tokenizer(review, padding='max_length',
                                        max_length=512, truncation=True,
                                        return_tensors="pt")
        input_ids = encoded_review['input_ids'].squeeze(0)
        attention_mask = encoded_review['attention_mask'].squeeze(0)

        return input_ids, attention_mask, target
