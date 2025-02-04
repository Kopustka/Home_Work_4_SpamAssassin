import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def read_email(file_path):
    """
    Reads an email file and returns its content as a string.

    Parameters:
        file_path (str): The path to the email file.

    Returns:
        str: The content of the email.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return content


def lemmatization_and_parsing(sample_spam, spam_files, ham_files):
    """
    Performs text preprocessing, including:
    - Removing HTML tags
    - Removing email addresses
    - Removing links and numbers
    - Converting text to lowercase
    - Removing punctuation
    - Removing stopwords
    - Applying lemmatization

    Then, it vectorizes the cleaned text using TF-IDF.

    Parameters:
        sample_spam (str): A sample spam email to test text cleaning.
        spam_files (list): List of file paths for spam emails.
        ham_files (list): List of file paths for ham (non-spam) emails.

    Returns:
        tuple: (TF-IDF vectorizer, cleaned corpus, TF-IDF transformed matrix)
    """

    # Download necessary NLTK resources
    nltk.download("stopwords")
    nltk.download("wordnet")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        """
        Cleans the given text by removing unnecessary elements and applying lemmatization.

        Parameters:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.
        """
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = re.sub(r"\S*@\S*\s?", "", text)  # Remove email addresses
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"\d+", "", text)  # Remove digits
        text = text.lower()  # Convert to lowercase
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)

    # Test text cleaning on a sample spam email
    print('CLEAN TEXT EXAMPLE:')
    print(clean_text(sample_spam))

    # Initialize TF-IDF vectorizer (limit vocabulary size for efficiency)
    vectorizer = TfidfVectorizer(max_features=5000)

    # Read, clean, and vectorize all emails
    corpus = [clean_text(read_email(f)) for f in spam_files + ham_files]
    X = vectorizer.fit_transform(corpus)

    return vectorizer, corpus, X


def creating_marks(spam_files, ham_files):
    """
    Creates labels for classification:
    - 1 for spam emails
    - 0 for ham (non-spam) emails

    Parameters:
        spam_files (list): List of file paths for spam emails.
        ham_files (list): List of file paths for ham emails.

    Returns:
        list: A list of labels (1 for spam, 0 for ham).
    """
    return [1] * len(spam_files) + [0] * len(ham_files)


def vectorizing(vectorizer, y, X):
    """
    Saves the vectorized data and labels using joblib.

    Parameters:
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        y (list): The classification labels (1 for spam, 0 for ham).
        X (sparse matrix): The TF-IDF transformed text data.

    Returns:
        None
    """
    joblib.dump(vectorizer, "../vectorizer.pkl")  # Save TF-IDF vectorizer
    joblib.dump(y, "../labels.pkl")  # Save classification labels
    joblib.dump(X, "../vectorized_data.pkl")  # Save transformed feature matrix
