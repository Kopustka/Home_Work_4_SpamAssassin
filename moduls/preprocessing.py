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
import joblib

def get_datas(spam_path, ham_path):
    # Получаем списки файлов
    spam_files = glob.glob(os.path.join(spam_path, "*"))
    ham_files = glob.glob(os.path.join(ham_path, "*"))

    return spam_files, ham_files


def balancing(spam_files, ham_files):
    # Вычисляем, сколько ham-файлов оставить
    # Определяем, сколько оставить ham-файлов
    target_ham_count = len(spam_files) + random.randint(30, 40)
    target_spam_count = len(ham_files) + random.randint(30, 40)

    if len(ham_files) > target_ham_count:
        ham_files = random.sample(ham_files, target_ham_count)

    # Балансируем spam-файлы (если их больше, случайно удаляем лишние)
    if len(spam_files) > target_spam_count:
        spam_files = random.sample(spam_files, target_spam_count)

    return spam_files, ham_files


def read_email(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return content






def lemmatization_and_parsing(sample_spam,spam_files,ham_files):
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

    return vectorizer, corpus, X


def creating_marks(spam_files, ham_files):
    # Создаем метки: 1 - спам, 0 - не спам
    y = [1] * len(spam_files) + [0] * len(ham_files)

    return y


def vectorizing(vectorizer, y, X):
    # Сохраняем векторайзер и векторизованные данные
    joblib.dump(vectorizer, "../vectorizer.pkl")
    joblib.dump(y, "../labels.pkl")
    joblib.dump(X, "../vectorized_data.pkl")

