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
    # ссылка в kaggle и путь установки
    # У меня почему-то даже без папки работает не знаю почему
    dataset_name = 'beatoa/spamassassin-public-corpus'
    download_path = '../data/'

    # путь до датасетов
    spam_path = '../data/spam_2/spam_2'
    ham_path = '../data/easy_ham/easy_ham'
    # Можно и '../data/hard_ham/hard_ham' но нужно подбирать настройки


    '''Importing'''
    print("------Importing------", end ='\n')

    importing(dataset_name,download_path)
    print(f"Датасет скачан в: {download_path}")


    '''Preprocessing'''
    print("----Preprocessing----", end ='\n')

    spam_files, ham_files = get_datas(spam_path, ham_path)
    print(f"Спам-файлов: {len(spam_files)}")
    print(f"Хам-файлов: {len(ham_files)}")

    # balancing
    spam_files, ham_files = balancing(spam_files, ham_files)
    print(f"После фильтрации: Спам: {len(spam_files)}, Хам: {len(ham_files)}")
    print(f"Оставшиеся хам-файлы: {len(ham_files)}")
    print(f"Оставшиеся спам-файлы: {len(spam_files)}")

    # Пример чтения первых писем
    sample_spam = read_email(spam_files[0])
    sample_ham = read_email(ham_files[0])
    print(sample_spam[:500])  # Первые 500 символов письма

    # настройка векторизатора
    vectorizer, corpus, X = lemmatization_and_parsing(sample_spam,spam_files,ham_files)


    # Preview
    y = creating_marks(spam_files, ham_files)

    # Vectorizing
    # Сохраняем векторайзер и векторизованные данные
    vectorizing(vectorizer, y, X)


    '''Traning'''
    print("------Traning------", end ='\n')

    # загрузка из векторов
    vectorizer, X, y = loading()
    # Получение тестовыой и тренировочной выборки
    X_train, X_test, y_train, y_test = preview(X,y)
    # предсказание, можно использовать:
    #                     randomforest,
    #                     gradientbosting,
    #                     svcclass(SVC),
    #                     extr_trees,
    #                     catboost,
    y_pred = svcclass(X_train, X_test, y_train)
    # Вывод результатов
    quality_assessment(y_test, y_pred)

if __name__ == "__main__":
    main()
