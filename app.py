from src.models.inference import BertInference
import streamlit as st


def main():
    st.title("Классификация текста при помощи ruBERT")

    model_path = "/home/nfrvnikita/projects/service4classification/notebooks/checkpoint/BertClassifier3.pt"
    tokenizer_path = "cointegrated/rubert-tiny"
    bert_inference = BertInference(model_path, tokenizer_path)

    text_input = st.text_area("Введите текст для классификации:")
    if st.button("Предсказать"):
        if text_input:
            predicted_class = bert_inference.predict(text_input)
            st.write(f"Предсказанный класс: {predicted_class}")


if __name__ == "__main__":
    main()
