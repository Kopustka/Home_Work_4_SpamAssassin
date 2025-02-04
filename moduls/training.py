from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib


def loading():
    """
    Loads the preprocessed and vectorized data from stored files.

    Returns:
        tuple: (vectorizer, X, y)
        - vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        - X (sparse matrix): The TF-IDF transformed feature matrix.
        - y (list): Classification labels (1 for spam, 0 for ham).
    """
    vectorizer = joblib.load("../vectorizer.pkl")
    X = joblib.load("../vectorized_data.pkl")
    y = joblib.load("../labels.pkl")
    return vectorizer, X, y


def preview(X, y):
    """
    Splits the dataset into training and testing sets.

    Parameters:
        X (sparse matrix): The TF-IDF transformed feature matrix.
        y (list): The classification labels.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        - X_train, X_test: Training and testing feature matrices.
        - y_train, y_test: Training and testing labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Feature matrix size:", X.shape)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def randomforest(X_train, X_test, y_train):
    """
    Trains and evaluates a Random Forest classifier.

    Parameters:
        X_train, X_test: Training and testing feature matrices.
        y_train: Training labels.

    Returns:
        y_pred: Predicted labels for the test set.
    """
    rf_model = RandomForestClassifier(n_estimators=90, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    return y_pred


def gradientbosting(X_train, X_test, y_train):
    """
    Trains and evaluates a Gradient Boosting classifier.

    Parameters:
        X_train, X_test: Training and testing feature matrices.
        y_train: Training labels.

    Returns:
        y_pred: Predicted labels for the test set.
    """
    gb_model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.13, max_depth=8, random_state=42)
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)

    return y_pred


def svcclass(X_train, X_test, y_train):
    """
    Trains and evaluates a Support Vector Machine (SVM) classifier.

    Parameters:
        X_train, X_test: Training and testing feature matrices.
        y_train: Training labels.

    Returns:
        y_pred: Predicted labels for the test set.
    """
    svc_model = SVC(kernel='rbf', C=1.0, gamma="scale")
    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_test)

    return y_pred


def extr_trees(X_train, X_test, y_train):
    """
    Trains and evaluates an Extra Trees classifier.

    Parameters:
        X_train, X_test: Training and testing feature matrices.
        y_train: Training labels.

    Returns:
        y_pred: Predicted labels for the test set.
    """
    et_model = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42)
    et_model.fit(X_train, y_train)

    y_pred = et_model.predict(X_test)

    return y_pred


def catboost(X_train, X_test, y_train):
    """
    Trains and evaluates a CatBoost classifier.

    Parameters:
        X_train, X_test: Training and testing feature matrices.
        y_train: Training labels.

    Returns:
        y_pred: Predicted labels for the test set.
    """
    et_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=100)
    et_model.fit(X_train, y_train)

    y_pred = et_model.predict(X_test)

    return y_pred


def quality_assessment(y_test, y_pred):
    """
    Evaluates the performance of the model using accuracy, classification report, and confusion matrix.

    Parameters:
        y_test: Actual labels for the test set.
        y_pred: Predicted labels by the model.

    Returns:
        None (prints evaluation results).
    """
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
