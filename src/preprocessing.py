import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_html(text):
    """Removes HTML tags and scripts."""
    text = re.sub(r'<(script|style).*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<.*?>', ' ', text)
    return text

def clean_text(text):
    """Full preprocessing pipeline for email text."""
    if not isinstance(text, str):
        return ""
        
    text = remove_html(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url_token ', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', ' email_token ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)