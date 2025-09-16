# Ниже представлен подробный пошаговый сценарий (скрипт) использования модели BERT для задач обработки естественного языка (например, классификация текста). 
# Этот скрипт основан на популярной библиотеке Hugging Face Transformers и включает все основные этапы: подготовку данных, токенизацию, обучение модели, оценку и использование модели для предсказаний. 

# 1. Установка необходимых библиотек
# pip install transformers datasets torch

# 2. Импорт необходимых модулей
import torch 
from transformer import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments 
from datasets import load_dataset, load_metric

# 3. Загрузка и подготовка данных Для примера возьмем датасет IMDb (отзывов о фильмах для задачи классификации).
# Загружаем датасет IMDb dataset = load_dataset("imdb") metric = load_metric("accuracy") 
# # Разделение данных на обучающую и тестовую выборки train_dataset = dataset['train'] test_dataset = dataset['test']

# 4. Инициализация токенизатора
# Инициализация токенизатора BERT (выберите базовую модель) 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 5. Токенизация и подготовка данных Создаем функции для токенизации с учетом ограниченной длины последовательности:
def tokenize_function(examples): 
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512) 
    # Токенизация данных 
    tokenized_train = train_dataset.map(tokenize_function, batched=True) 
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Переконвертируем данные в формат, совместимый с моделью:
    # Удалим лишние колонки и оставим только необходимые 
    columns_to_keep = ['input_ids', 'attention_mask', 'label'] 
    train_dataset = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col not in columns_to_keep]) 
    test_dataset = tokenized_test.remove_columns([col for col in tokenized_test.column_names if col not in columns_to_keep]) # Укажем формат данных для PyTorch train_dataset.set_format(type='torch', columns=columns_to_keep) test_dataset.set_format(type='torch', columns=columns_to_keep)

    # 6. Инициализация модели BERT для классификации
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    #7. Настройка параметров обучения
    training_args = TrainingArguments( output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8, per_device_eval_batch_size=8, evaluation_strategy='epoch', save_strategy='epoch', logging_dir='./logs', logging_steps=10, )
    
# 8. Определение метрик для оценки
def compute_metrics(eval_pred): 
    logits, labels = eval_pred 
    predictions = torch.argmax(torch.tensor(logits), dim=-1) 

    # 9. Обучение модели с помощью Trainer
    trainer = Trainer( model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics, ) trainer.train()
    10. Оценка модели
    # Оценка на тестовых данных results = trainer.evaluate() print(results)
    11. Использование модели для предсказаний
    # Пример текста 
    texts = ["This movie was fantastic!", "I did not like this film."] 
    # Токенизация 
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt') 
    # Предсказания 
    with torch.no_grad(): 
        outputs = model(**inputs) 
        logits = outputs.logits 
        predictions = torch.argmax(logits, dim=-1) 
        for text, pred in zip(texts, predictions): 
            label = 'Positive' if pred.item() == 1 else 'Negative' 
            print(f"Text: {text}\nPrediction: {label}\n")

# Итоги Этот скрипт демонстрирует полный процесс работы с моделью BERT для задачи классификации текста: 
# • установка и подготовка данных, 
# • токенизация, 
# • настройка модели, 
# • обучение и оценка, 
# • предсказания на новых данных. 
# Для других задач (например, Named Entity Recognition, вопросные ответные системы) структура будет похожая, только с изменениями в конфигурации модели и функциях обработки данных.
