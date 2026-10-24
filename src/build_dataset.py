import pandas as pd
import os

os.makedirs('data/processed', exist_ok=True)

def load_and_format(filepath, text_cols, label_map=None, label_col='label', default_label=None, encoding='utf-8'):
    """Loads a CSV, combines text columns, and maps labels."""
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    # Combine text columns (e.g., subject + body)
    if len(text_cols) > 1:
        df['text'] = df[text_cols[0]].fillna('') + " " + df[text_cols[1]].fillna('')
    else:
        df['text'] = df[text_cols[0]].fillna('')

    # Assign or map labels
    if default_label is not None:
        df['label'] = default_label
    else:
        df['label'] = df[label_col].map(label_map)

    return df[['text', 'label']]

print("Loading datasets...")

# 1. CEAS_08 (Spam) - label 1 is spam
ceas = load_and_format('data/raw/CEAS_08.csv', ['subject', 'body'], {1: 1})

# 2. Enron (Legitimate) - label 0 is legit
enron = load_and_format('data/raw/Enron.csv', ['subject', 'body'], {0: 0})

# 3. Ling (Legit/Spam)
ling = load_and_format('data/raw/Ling.csv', ['subject', 'body'], {0: 0, 1: 1})

# 4. Nazario (Phishing) - Force to Phishing (2)
nazario = load_and_format('data/raw/Nazario.csv', ['subject', 'body'], {1: 2, 0: 2})

# 5. Nigerian_Fraud (Phishing) - Force to Phishing (2)
nigeria = load_and_format('data/raw/Nigerian_Fraud.csv', ['subject', 'body'], {1: 2})

# 6. SpamAssasin (Spam/Legit)
spamassasin = load_and_format('data/raw/SpamAssasin.csv', ['subject', 'body'], {1: 1, 0: 0})

# 7. phishing_email (Phishing/Legit) - label 1 is phishing
phish_email = load_and_format('data/raw/phishing_email.csv', ['text_combined'], {1: 2, 0: 0})

# 8. spam.csv (SMS Spam - v1 is label, v2 is text)
spam_txt = load_and_format('data/raw/spam.csv', ['v2'], {'ham': 0, 'spam': 1}, label_col='v1')

# 9. emails.csv (Huge Enron dump - 517k rows) 
# FIX: No label column, so we force it to 0 (Legitimate).
# We also take a random sample of 20,000 rows to keep training fast!
print("Sampling 20,000 rows from emails.csv...")
raw_enron = load_and_format('data/raw/emails.csv', ['message'], default_label=0)
raw_enron = raw_enron.sample(n=20000, random_state=42)

# Combine all into one dataframe
print("Combining all datasets...")
master_df = pd.concat([ceas, enron, ling, nazario, nigeria, spamassasin, phish_email, spam_txt, raw_enron], ignore_index=True)

# Drop rows with empty text
master_df = master_df.dropna(subset=['text'])
master_df = master_df[master_df['text'].str.strip() != '']

# Drop duplicates
master_df = master_df.drop_duplicates(subset=['text'])

# Show results
print(f"\nMaster Dataset Shape: {master_df.shape}")
print("\nClass Distribution (0=Legit, 1=Spam, 2=Phishing):")
print(master_df['label'].value_counts())

# Save
master_df.to_csv('data/processed/master_dataset.csv', index=False)
print("\nSuccessfully saved to data/processed/master_dataset.csv")