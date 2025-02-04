from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

def loading():
    vectorizer = joblib.load("../vectorizer.pkl")
    X = joblib.load("../vectorized_data.pkl")
    y = joblib.load("../labels.pkl")
    return vectorizer, X, y

def preview(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Размер матрицы признаков:", X.shape)
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def randomforest(X_train, X_test, y_train):
    rf_model = RandomForestClassifier(n_estimators=90, random_state=42)  # 100 деревьев
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    return y_pred


def gradientbosting(X_train, X_test, y_train):
    gb_model = GradientBoostingClassifier(n_estimators=120,learning_rate=0.13,max_depth=8, random_state=42)
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)

    return y_pred


def svcclass(X_train, X_test, y_train):
    svc_model = SVC(kernel='rbg', C=1.0, gamma="scale")
    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_test)

    return y_pred


def extr_trees(X_train, X_test, y_train):
    et_model = ExtraTreesClassifier(n_estimators=200,max_depth=15, random_state=42)
    et_model.fit(X_train, y_train)

    y_pred = et_model.predict(X_test)

    return y_pred

def catboost(X_train, X_test, y_train):
    et_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=100)
    et_model.fit(X_train, y_train)

    y_pred = et_model.predict(X_test)

    return y_pred


def quality_assessment(y_test, y_pred):
    # Оцениваем качество
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy:.4f}")

    # Детальный отчёт
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))