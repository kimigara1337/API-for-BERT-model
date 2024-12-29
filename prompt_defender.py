import torch
from transformers import BertTokenizer, BertForSequenceClassification


class PromptDefenderClassifier:
    def __init__(self):
        # Устройство для выполнения
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка модели и токенизатора
        self.model = BertForSequenceClassification.from_pretrained("my_bert_model")
        self.tokenizer = BertTokenizer.from_pretrained("my_bert_model")
        self.model.to(self.device)
        self.model.eval()  # Переключаем модель в режим оценки

    def check_on_bad_request(self, prompt_input):
        # Токенизация нового текста
        new_texts_tokenized = self.tokenizer(
            [prompt_input],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        # Предсказания
        with torch.no_grad():
            input_ids = new_texts_tokenized["input_ids"].to(self.device)
            attention_mask = new_texts_tokenized["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Возвращаем результат
        return int(predictions.cpu().numpy()[0])
