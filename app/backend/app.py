# Change line 1 to this:
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import sys
import os

# Add src to path so we can import clean_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from preprocessing import clean_text

app = Flask(__name__)
CORS(app) # Allows the frontend to talk to the backend

# 👇 ADD THIS NEW ROUTE 👇
@app.route('/')
def home():
    # This points Flask to the frontend folder
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))
    return send_from_directory(frontend_dir, 'index.html')
# 👆 END OF NEW ROUTE 👆# Allows the frontend to talk to the backend

# Load the saved model and vectorizer
MODEL_PATH = 'models/best_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

print("Loading models...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("Models loaded successfully!")

# Map numbers back to text
LABELS = {
    0: "Legitimate",
    1: "Spam",
    2: "Phishing"
}

def softmax(x):
    """Calculate confidence score from SVM decision function."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email_text = data.get('text', '')
        
        if not email_text.strip():
            return jsonify({'error': 'No text provided'}), 400

        # 1. Clean the text
        cleaned_text = clean_text(email_text)
        
        # 2. Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = model.predict(vectorized_text)[0]
        
        # 4. Get confidence score (Softmax on SVM decision function)
        decision_scores = model.decision_function(vectorized_text)[0]
        probabilities = softmax(decision_scores)
        confidence = float(np.max(probabilities)) * 100

        # 5. Return result
        result = {
            'prediction': LABELS.get(int(prediction), "Unknown"),
            'confidence': round(confidence, 2)
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(debug=True, port=5000)