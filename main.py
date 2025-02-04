# Import necessary modules from the 'moduls' package
from moduls import importing
from moduls import (get_datas,
                    balancing,
                    read_email,
                    lemmatization_and_parsing,
                    creating_marks,
                    vectorizing)
from moduls import (loading,
                    preview,
                    randomforest,
                    gradientbosting,
                    svcclass,
                    extr_trees,
                    catboost,
                    quality_assessment
                    )

def main():
    # Kaggle dataset name and download path
    # For some reason, it works even without specifying a folder
    dataset_name = 'beatoa/spamassassin-public-corpus'
    download_path = '../data/'

    # Paths to spam and ham datasets
    spam_path = '../data/spam_2/spam_2'
    ham_path = '../data/easy_ham/easy_ham'
    # '../data/hard_ham/hard_ham' can also be used, but parameters need adjustment

    '''Importing'''
    print("------Importing------", end ='\n')

    # Download the dataset from Kaggle
    importing(dataset_name, download_path)
    print(f"Dataset downloaded to: {download_path}")

    '''Preprocessing'''
    print("----Preprocessing----", end ='\n')

    # Get the list of spam and ham files from the specified directories
    spam_files, ham_files = get_datas(spam_path, ham_path)
    print(f"Spam files: {len(spam_files)}")
    print(f"Ham files: {len(ham_files)}")

    # Balance the dataset by randomly removing excess spam or ham files
    spam_files, ham_files = balancing(spam_files, ham_files)
    print(f"After balancing: Spam: {len(spam_files)}, Ham: {len(ham_files)}")
    print(f"Remaining ham files: {len(ham_files)}")
    print(f"Remaining spam files: {len(spam_files)}", end='\n')

    # Read a sample email from spam and ham datasets
    sample_spam = read_email(spam_files[0])
    sample_ham = read_email(ham_files[0])
    print('MAIL EXAMPLE:')
    print(sample_spam[:500], end='\n')  # Display the first 500 characters of a spam email

    # Preprocess the dataset (lemmatization, tokenization, and text parsing)
    vectorizer, corpus, X = lemmatization_and_parsing(sample_spam, spam_files, ham_files)

    # Create labels (1 for spam, 0 for ham) for classification
    y = creating_marks(spam_files, ham_files)

    # Vectorize the dataset and save the transformed data for later use
    vectorizing(vectorizer, y, X)

    '''Training'''
    print("------Training------", end ='\n')

    # Load the saved vectorized data
    vectorizer, X, y = loading()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = preview(X, y)

    # Train a classification model and make predictions
    # Available models:
    #     randomforest        -> Random Forest Classifier
    #     gradientbosting     -> Gradient Boosting Classifier
    #     svcclass(SVC)       -> Support Vector Machine (SVM)
    #     extr_trees          -> Extra Trees Classifier
    #     catboost            -> CatBoost Classifier
    y_pred = gradientbosting(X_train, X_test, y_train)  # Using Gradient Boosting

    # Evaluate the model performance with accuracy, precision, recall, and confusion matrix
    quality_assessment(y_test, y_pred)

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
