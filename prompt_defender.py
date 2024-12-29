import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch

import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='application.log'  # Вывод в файл (или можно убрать, чтобы логи шли в консоль)
)



class PromtDefenderClassifier():
    def __init__(self):
        pass

    def data_read():
        # Чтение файлов
        self.jailbreak_prompts_2023_05_07 = pd.read_csv('jailbreak_llms/data/prompts/jailbreak_prompts_2023_05_07.csv', sep=',', encoding='utf-8')
        self.jailbreak_prompts_2023_12_25 = pd.read_csv('jailbreak_llms/data/prompts/jailbreak_prompts_2023_12_25.csv', sep=',', encoding='utf-8')
        self.regular_prompts_2023_05_07 = pd.read_csv('jailbreak_llms/data/prompts/regular_prompts_2023_05_07.csv', sep=',', encoding='utf-8')
        self.regular_prompts_2023_12_25 = pd.read_csv('jailbreak_llms/data/prompts/regular_prompts_2023_12_25.csv', sep=',', encoding='utf-8')

        # Удаление лишних столбцов
        self.jailbreak_prompts_2023_05_07 = self.jailbreak_prompts_2023_05_07.drop(columns=['community_id', 'community_name'], errors='ignore')
        self.jailbreak_prompts_2023_12_25 = self.jailbreak_prompts_2023_12_25.drop(columns=['community', 'community_id', 'previous_community_id'], errors='ignore')

        # # Dывод первых строк
        # print("jailbreak_prompts_2023_05_07.csv")
        # display(jailbreak_prompts_2023_05_07.head())

        # print("\njailbreak_prompts_2023_12_25.csv")
        # display(jailbreak_prompts_2023_12_25.head())

        # print("\nregular_prompts_2023_05_07.csv")
        # display(regular_prompts_2023_05_07.head())

        # print("\nregular_prompts_2023_12_25.csv")
        # display(regular_prompts_2023_12_25.head())

    def data_preparate_and_fit(self):
        # Добавляем метки
        self.jailbreak_prompts_2023_05_07['label'] = 1
        self.jailbreak_prompts_2023_12_25['label'] = 1
        self.regular_prompts_2023_05_07['label'] = 0
        self.regular_prompts_2023_12_25['label'] = 0

        # Объединяем все датасеты в один
        all_data = pd.concat([self.jailbreak_prompts_2023_05_07, self.jailbreak_prompts_2023_12_25, self.regular_prompts_2023_05_07, self.regular_prompts_2023_12_25], ignore_index=True)

        # Сохраняем объединённый датасет для дальнейшего использования
        all_data.to_csv('/content/combined_prompts.csv', index=False)

        # print("Combined data sample:")
        # display(all_data.head())


        # Разделяем данные на обучающую и тестовую выборки
        self.X = all_data['prompt']  # Промпты (вопросы)
        self.y = all_data['label']  # Метки (джейлбрек/обычные)

        
        # По идее в этом смысла нет. Но сокорее так будет удобнее чтобы кор подстраивать по быстрому. 
        self.X_train, self.X_test, self.y_train, self.y_test = self.X, self.X, self.y, self.y

        # Разделяем на 80% для обучения и 20% для тестирования # т.к. прод вариант оно отсутствует. Хотим обучаться на всх данных.
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # print(f"Обучающая выборка: {len(X_train)} запросов")
        # print(f"Тестовая выборка: {len(X_test)} запросов")


        # Преобразуем в PyTorch тензоры
        train_input_ids_tensor = torch.tensor(train_input_ids.numpy())
        train_attention_mask_tensor = torch.tensor(train_attention_mask.numpy())
        train_labels_tensor = torch.tensor(y_train.values)  # предполагаю, что y_train — это ваши метки для обучения

        test_input_ids_tensor = torch.tensor(test_input_ids.numpy())
        test_attention_mask_tensor = torch.tensor(test_attention_mask.numpy())
        test_labels_tensor = torch.tensor(y_test.values)  # предполагаю, что y_test — это ваши метки для теста

        # Создаем TensorDataset для тренировки и теста
        train_dataset = TensorDataset(train_input_ids_tensor, train_attention_mask_tensor, train_labels_tensor)
        test_dataset = TensorDataset(test_input_ids_tensor, test_attention_mask_tensor, test_labels_tensor)

        # Создаем DataLoader для батчей
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64)





        # Кастомный датасет
        class CustomDataset(Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx]
                }

        # Преобразуем в PyTorch тензоры
        train_input_ids_tensor = torch.tensor(train_input_ids.numpy())
        train_attention_mask_tensor = torch.tensor(train_attention_mask.numpy())
        train_labels_tensor = torch.tensor(y_train.values)

        test_input_ids_tensor = torch.tensor(test_input_ids.numpy())
        test_attention_mask_tensor = torch.tensor(test_attention_mask.numpy())
        test_labels_tensor = torch.tensor(y_test.values)

        # Создаем кастомные датасеты
        train_dataset = CustomDataset(train_input_ids_tensor, train_attention_mask_tensor, train_labels_tensor)
        test_dataset = CustomDataset(test_input_ids_tensor, test_attention_mask_tensor, test_labels_tensor)


        # Преобразование меток в список перед созданием тензоров
        y_train_tensor = torch.tensor(y_train.values.tolist())
        y_test_tensor = torch.tensor(y_test.values.tolist())

        # Создание TensorDataset для обучения и тестирования
        train_dataset = TensorDataset(
            X_train_tokenized['input_ids'], 
            X_train_tokenized['attention_mask'], 
            y_train_tensor
        )

        test_dataset = TensorDataset(
            X_test_tokenized['input_ids'], 
            X_test_tokenized['attention_mask'], 
            y_test_tensor
        )


        # Преобразуем метки в PyTorch тензоры, если они pandas.Series
        y_train = torch.tensor(y_train.values) if isinstance(y_train, pd.Series) else torch.tensor(y_train)
        y_test = torch.tensor(y_test.values) if isinstance(y_test, pd.Series) else torch.tensor(y_test)

        # Токенизация
        X_train_tokenized = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=512)
        X_test_tokenized = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt', max_length=512)

        # Создание датасетов
        train_dataset = TensorDataset(X_train_tokenized['input_ids'], X_train_tokenized['attention_mask'], y_train)
        test_dataset = TensorDataset(X_test_tokenized['input_ids'], X_test_tokenized['attention_mask'], y_test)

        print(f"Len X_train_tokenized: {len(X_train_tokenized['input_ids'])}, Len y_train: {len(y_train)}")
        print(f"Len X_test_tokenized: {len(X_test_tokenized['input_ids'])}, Len y_test: {len(y_test)}")

        print(X_train.isnull().sum())  # Для Pandas DataFrame или Series
        print(y_train.isnull().sum())
        print(X_test.isnull().sum())
        print(y_test.isnull().sum())



        # Загружаем токенизатор и модель
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # Токенизация
        X_train_tokenized = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=512)
        X_test_tokenized = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt', max_length=512)

        # Создание DataLoader для обучения и тестирования
        train_dataset = TensorDataset(X_train_tokenized['input_ids'], X_train_tokenized['attention_mask'], torch.tensor(y_train))
        test_dataset = TensorDataset(X_test_tokenized['input_ids'], X_test_tokenized['attention_mask'], torch.tensor(y_test))

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64)

        # Определяем оптимизатор
        optimizer = AdamW(model.parameters(), lr=1e-5)

        # Устройство для обучения (GPU или CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Функция для обучения модели
        def train_epoch(model, dataloader, optimizer, device):
            model.train()
            total_loss = 0
            for batch in dataloader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                optimizer.zero_grad()

                # Прямой проход
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Обратное распространение ошибки
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Training loss: {avg_loss}")
            return avg_loss

        # Функция для оценки модели
        def eval_epoch(model, dataloader, device):
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for batch in dataloader:
                    input_ids, attention_mask, labels = [x.to(device) for x in batch]
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)

            accuracy = total_correct / total_samples
            print(f"Validation accuracy: {accuracy}")
            return accuracy

        # Цикл обучения
        num_epochs = 3
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Обучение на одном эпохе
            train_loss = train_epoch(model, train_dataloader, optimizer, device)

            # Оценка на валидационном наборе
            eval_accuracy = eval_epoch(model, test_dataloader, device)

        print("Обучение завершено")


        #Сохраняем модель
        model.save_pretrained("my_bert_model")
        tokenizer.save_pretrained("my_bert_model")



    def check_on_bed_request(self, promt_input):


        # Устройство
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка модели и токенизатора
        model = BertForSequenceClassification.from_pretrained("my_bert_model")
        tokenizer = BertTokenizer.from_pretrained("my_bert_model")
        model.to(device)

        # Переключение в режим оценки (чтобы случайно не начать обучение)
        model.eval()

        # Пример использования
        new_texts = [
            promt_input
            # "The product quality is excellent!", #Пример 1
            # "I hate the experience. Very disappointing!"  # Пример 2
        ]

        # Токенизация новых текстов
        new_texts_tokenized = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

        # Предсказания
        with torch.no_grad():
            input_ids = new_texts_tokenized['input_ids'].to(device)
            attention_mask = new_texts_tokenized['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Результат
        print("Predictions:", predictions.cpu().numpy())
        resalt = predictions.cpu().numpy()
        return str(resalt)
        #Далее мы развернем Llama и протестируем нашу обученную BERT 
