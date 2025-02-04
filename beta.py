import os
import glob
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
import kagglehub

# Запустить установку а потом уже запускать весь код
path = kagglehub.dataset_download("beatoa/spamassassin-public-corpus")

print("Path to dataset files:", path)


spam_path = 'spam_2'
ham_path = 'easy_ham'

# Получаем списки файлов
spam_files = glob.glob(os.path.join(spam_path, "*"))
ham_files = glob.glob(os.path.join(ham_path, "*"))

print(f"Спам-файлов: {len(spam_files)}")
print(f"Хам-файлов: {len(ham_files)}")

# Вычисляем, сколько ham-файлов оставить
# Определяем, сколько оставить ham-файлов
target_ham_count = len(spam_files) + random.randint(30, 40)

if len(ham_files) > target_ham_count:
    # Случайно удаляем лишние элементы из списка (но не из папки!)
    ham_files = random.sample(ham_files, target_ham_count)

print(f"После фильтрации: Спам: {len(spam_files)}, Хам: {len(ham_files)}")

# Итоговое количество файлов
print(f"Оставшиеся хам-файлы: {len(ham_files)}")
print(f"Оставшиеся спам-файлы: {len(spam_files)}")



def read_email(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return content

# Пример чтения первых писем
sample_spam = read_email(spam_files[0])
sample_ham = read_email(ham_files[0])

print(sample_spam[:500])  # Первые 500 символов письма


nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Убираем HTML
    text = re.sub(r"\S*@\S*\s?", "", text)  # Удаляем email-адреса
    text = re.sub(r"http\S+", "", text)  # Удаляем ссылки
    text = re.sub(r"\d+", "", text)  # Удаляем цифры
    text = text.lower()  # Приводим к нижнему регистру
    text = text.translate(str.maketrans("", "", string.punctuation))  # Убираем пунктуацию
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Проверяем очистку
print(clean_text(sample_spam))



# Векторизатор TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Ограничиваем словарь для скорости
corpus = [clean_text(read_email(f)) for f in spam_files + ham_files]
X = vectorizer.fit_transform(corpus)

# Создаем метки: 1 - спам, 0 - не спам
y = [1] * len(spam_files) + [0] * len(ham_files)

print("Размер матрицы признаков:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

import joblib

# Сохраняем векторайзер и векторизованные данные
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(X, "vectorized_data.pkl")
joblib.dump(y, "labels.pkl")