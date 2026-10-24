# Enterprise Email Threat Detection Platform

An AI-powered cybersecurity platform that classifies emails into three categories: **Legitimate, Spam, and Phishing**. 

## 🚀 Features
- **Machine Learning Pipeline:** NLP preprocessing (Tokenization, Lemmatization, Stop-word removal) and TF-IDF vectorization.
- **Model Comparison:** Trained and evaluated Naive Bayes, Logistic Regression, Linear SVM, and Random Forest.
- **Best Model:** Linear SVM achieved **96.77% Accuracy** and **95.75% Macro F1-Score**.
- **Full-Stack Application:** Flask API backend connected to a responsive HTML/CSS/JS frontend dashboard.

## 📁 Project Structure
- `data/` - Raw datasets and processed `master_dataset.csv`
- `models/` - Saved best model (`best_model.pkl`) and vectorizer (`tfidf_vectorizer.pkl`)
- `src/` - Source code for preprocessing, dataset building, and model training
- `app/backend/` - Flask API server
- `app/frontend/` - Web dashboard

## 🛠️ How to Run
1. **Activate Environment:** `source .venv/Scripts/activate` (Windows)
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Start Backend:** `python app/backend/app.py`
4. **Open Dashboard:** Double-click `app/frontend/index.html` to open in your browser.

## 📊 Results
| Model | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| Naive Bayes | 84.65% | 78.10% |
| Logistic Regression | 96.19% | 95.08% |
| **Linear SVM (Best)** | **96.77%** | **95.75%** |
| Random Forest | 96.20% | 94.57% |