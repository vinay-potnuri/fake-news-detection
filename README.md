# Fake News Detection (NLP Project)

## 📌 Project Goal
Build a machine learning model to classify news articles as **fake** or **real** using NLP techniques.

## 📂 Dataset
- Fake and Real News dataset (Kaggle)
- Two CSV files: `Fake.csv` and `True.csv`
- Combined into one dataframe with a `label` column:
  - 0 = Fake news
  - 1 = Real news

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

## 🔍 Approach
1. Load `Fake.csv` and `True.csv`
2. Add labels (0 for fake, 1 for real)
3. Combine datasets and shuffle
4. Keep only `text` and `label` columns
5. Train-test split
6. Convert text to features using **TF-IDF**
7. Train **Logistic Regression** model
8. Evaluate using:
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix

## 📊 Model Performance
(Add your actual numbers here after training, for example:)
- Accuracy: ~0.97
- High precision and recall for both classes

## 📁 Files in this Repository
- `fake_news_detection.ipynb` – complete Jupyter notebook with data loading, preprocessing, training and evaluation
- `README.md` – project description

## 🚀 Possible Improvements
- Try other models (SVM, Random Forest, Naive Bayes)
- Use word embeddings or transformers (BERT)
- Build a Streamlit app where user can paste an article and get prediction
