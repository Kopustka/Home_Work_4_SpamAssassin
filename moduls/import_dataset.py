from kaggle.api.kaggle_api_extended import KaggleApi

# Настройка Kaggle API
api = KaggleApi()
api.authenticate()


def importing(dataset_name,download_path):
    # Скачиваем
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

