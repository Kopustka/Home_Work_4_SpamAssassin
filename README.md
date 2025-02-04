Homework #4 - task description: 
Implement a minimum of 5 classifiers, compare metrics against each other, choose the best one for your dataset.
Classifiers(Used):
✓ Gradient Busting Classifier. 
✓ CatBoost classifier.
✓ Extra Trees classifier.
✓ Decision Tree classifier.
✓ SVM linear kernel.

Dataset:
11. (My dataset)Use the SpamAssassin dataset to train a model to classify email as spam or non-spam. Compare the results using different algorithms.


Problem Solution
📌 Spam Classifier using Machine Learning
This project trains machine learning models to classify emails into spam and non-spam using the SpamAssassin dataset.
The code includes data preprocessing, class balancing, text vectorization, and training multiple models.

📂 Project structure
bash
Copy
Edit
📁 project_folder/  
│─── 📁 moduls/ # Modules for import, processing, training  
│ │ │──── import_dataset.py # Load dataset from Kaggle  
│ │ │──── preprocessing.py # Text cleanup, lemmatization  
│ │ │──── balancing.py # Balancing classes  
│ │ │──── training.py # Training models  
│ │ │──── utils.py # Additional features  
│─── 📁 data/ # Folder to store the loaded dataset  
│──── main.py # Main startup file  
│──── requirements.txt # List of dependencies  
│──── README.md # Documentation  
🚀 Install and run
1️⃣ Install dependencies
Install all libraries before running:

bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Configuring the Kaggle API
You need an API key to download the dataset.

Go to the Kaggle API and download kaggle.json.
Put it in the ~/.kaggle/ (Linux/macOS) or C:\Users\USERNAME\.kaggle\ (Windows) folder.
Make sure the API is working:
bash
Copy
Edit
kaggle datasets list
3️⃣ Start the project
After installing everything you need, just run:

bash
Copy
Edit
python main.py
🛠️ Main code steps
🔹 1. Data download (import_dataset.py)
The file downloads the SpamAssassin dataset from Kaggle:

python
Copy
Edit
importing(dataset_name, download_path)
🔹 2. Preprocessing(preprocessing.py)
Loads spam and ham files
Removes extra characters, HTML, stop words
Applies TF-IDF for text vectorization
🔹 3. Class balancing (balancing.py)
If ham > spam → accidentally deletes extra ham files
If spam > ham → randomly delete unnecessary spam files
Uses SMOTE if classes are highly unequal
🔹 4. Model training (training.py)
5 models are available:

python
Copy
Edit
y_pred = randomforest(X_train, X_test, y_train) # Random Forest  
y_pred = gradientbosting(X_train, X_test, y_train) # Gradient Boosting  
y_pred = svcclass(X_train, X_test, y_train) # SVM  
y_pred = extr_trees(X_train, X_test, y_train) # Extra Trees  
y_pred = catboost(X_train, X_test, y_train) # CatBoost  
✅ You can choose any model you want!

🔹 5. Quality_assessment().
After prediction, the program shows:

Accuracy
Precision, Recall, F1-score
Confusion Matrix
📦 Dependencies (requirements.txt)
If you need to manually install libraries, add to requirements.txt:

beautifulsoup4==4.13.1
bleach==6.2.0
bs4==0.0.2
catboost==1.2.7
certifi==2024.12.14
charset-normalizer==3.4.1
click==8.1.8
colorama==0.4.6
contourpy==1.3.1
cycler==0.12.1
fonttools==4.55.8
graphviz==0.20.3
idna==3.10
joblib==1.4.2
kaggle==1.6.17
kagglehub==0.3.6
kiwisolver==1.4.8
matplotlib==3.10.0
narwhals==1.25.0
nltk==3.9.1
numpy==1.26.4
packaging==24.2
pandas==2.2.3
pillow==11.1.0
plotly==6.0.0
pyparsing==3.2.1
python-dateutil==2.9.0.post0
python-slugify==8.0.4
pytz==2025.1
regex==2024.11.6
requests==2.32.3
scikit-learn==1.6.1
scipy==1.15.1
six==1.17.0
soupsieve==2.6
text-unidecode==1.3
threadpoolctl==3.5.0
tqdm==4.67.1
typing_extensions==4.12.2
tzdata==2025.1
urllib3==2.3.0
webencodings==0.5.1


✨ Done! Now you can run and improve the spam classifier! 🚀
If you have any questions - write! 😊

Translated with DeepL.com (free version)

