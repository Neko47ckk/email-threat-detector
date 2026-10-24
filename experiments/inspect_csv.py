import pandas as pd

files_to_check = [
    'data/raw/Enron.csv', 
    'data/raw/Ling.csv', 
    'data/raw/phishing_email.csv', 
    'data/raw/Nigerian_Fraud.csv', 
    'data/raw/SpamAssasin.csv',
    'data/raw/spam.csv'
]

for f in files_to_check:
    try:
        # Try reading with default utf-8 first
        df = pd.read_csv(f)
    except UnicodeDecodeError:
        # If it fails, use latin-1
        df = pd.read_csv(f, encoding='latin-1')
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

    print(f"\n--- {f} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(1))