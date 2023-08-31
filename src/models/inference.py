import torch
from src.data.preprocessing import text_preprocessing
from transformers import BertTokenizer


class BertInference:
    def __init__(self, model_path, tokenizer_path):
        self.model = torch.load(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, text):
        text = text_preprocessing(text)  # Apply your preprocessing function
        encoded_text = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask)

        predicted_class = output.logits.argmax(dim=1).item()
        return predicted_class
