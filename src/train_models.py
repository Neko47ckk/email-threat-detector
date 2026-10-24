import pandas as pd
import joblib
import os
import time
import sys

# Import our custom preprocessing function
sys.path.append('src')
from preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def train_and_evaluate():
    print("1. Loading master dataset...")
    df = pd.read_csv('data/processed/master_dataset.csv')
    
    # We drop NAs just in case
    df = df.dropna(subset=['text', 'label'])
    
    print(f"Total emails to process: {len(df)}")
    print("2. Cleaning text (this may take a few minutes)...")
    
    # Apply the cleaning function
    start_time = time.time()
    df['clean_text'] = df['text'].apply(clean_text)
    print(f"Cleaning completed in {round((time.time() - start_time)/60, 2)} minutes.")
    
    # Drop any rows that became empty after cleaning
    df = df[df['clean_text'].str.strip() != '']

    print("3. Splitting data into Train and Test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    print("4. Extracting TF-IDF Features...")
    # Max features=10000 keeps the model fast
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save the vectorizer for the Flask app
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    # 5. Define the models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Linear SVM": LinearSVC(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    }

    results = {}
    best_f1 = 0
    best_model_name = ""
    best_model = None

    print("\n5. Training and Evaluating Models...\n" + "="*50)
    
    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        
        results[name] = {"Accuracy": acc, "Macro F1": macro_f1, "Time": round(time.time() - start, 2)}
        
        print(f"--- {name} Results ---")
        print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        print(classification_report(y_test, predictions, target_names=['Legit', 'Spam', 'Phishing']))
        print("="*50)
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_name = name
            best_model = model

    # 6. Save the best model
    print(f"\n🏆 Best Model: {best_model_name} with Macro F1: {best_f1:.4f}")
    joblib.dump(best_model, 'models/best_model.pkl')
    print("Model saved to models/best_model.pkl")
    
    # Print a final summary table
    print("\n--- Final Model Comparison ---")
    summary_df = pd.DataFrame(results).T
    print(summary_df)

if __name__ == "__main__":
    train_and_evaluate()