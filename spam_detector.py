import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
import joblib

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=stopwords.words('english'),
            lowercase=True,
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
        self.model = MultinomialNB(alpha=0.1)  # Add smoothing parameter
        
    def preprocess_text(self, text):
        """Preprocess text by removing special characters, converting to lowercase, etc."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, df):
        """Extract features from email data."""
        # Combine subject and body
        df['combined_text'] = df['subject'] + ' ' + df['body']
        
        # Preprocess text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(df['processed_text'])
        
        # Additional features
        df['has_link'] = df['combined_text'].str.contains(r'http[s]?://', regex=True).astype(int)
        df['text_length'] = df['processed_text'].str.len()
        df['has_urgent'] = df['subject'].str.contains(r'urgent|important|alert', case=False).astype(int)
        df['has_offer'] = df['combined_text'].str.contains(r'offer|deal|discount|free', case=False).astype(int)
        
        # Combine all features
        additional_features = df[['has_link', 'text_length', 'has_urgent', 'has_offer']].values
        features = np.hstack([tfidf_features.toarray(), additional_features])
        
        return features
    
    def train(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict_single_email(self, subject, body):
        """Predict whether a single email is spam or ham."""
        # Create a DataFrame with the input
        df = pd.DataFrame({
            'subject': [subject],
            'body': [body]
        })
        
        # Extract features
        df['combined_text'] = df['subject'] + ' ' + df['body']
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Transform text using the trained vectorizer
        tfidf_features = self.vectorizer.transform(df['processed_text'])
        
        # Extract additional features
        has_link = df['combined_text'].str.contains(r'http[s]?://', regex=True).astype(int)
        text_length = df['processed_text'].str.len()
        has_urgent = df['subject'].str.contains(r'urgent|important|alert', case=False).astype(int)
        has_offer = df['combined_text'].str.contains(r'offer|deal|discount|free', case=False).astype(int)
        
        # Combine features
        additional_features = np.array([has_link, text_length, has_urgent, has_offer]).T
        features = np.hstack([tfidf_features.toarray(), additional_features])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return prediction, probability

def main():
    # Try to load Enron dataset first
    try:
        # Load ham emails
        ham_dir = 'dataset/enron/ham'
        ham_emails = []
        for filename in os.listdir(ham_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1') as f:
                    content = f.read()
                    # Split into subject and body (first line is usually subject)
                    lines = content.split('\n', 1)
                    subject = lines[0] if lines else ''
                    body = lines[1] if len(lines) > 1 else ''
                    ham_emails.append({
                        'subject': subject,
                        'body': body,
                        'label': 'ham'
                    })
        
        # Load spam emails
        spam_dir = 'dataset/enron/spam'
        spam_emails = []
        for filename in os.listdir(spam_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1') as f:
                    content = f.read()
                    # Split into subject and body
                    lines = content.split('\n', 1)
                    subject = lines[0] if lines else ''
                    body = lines[1] if len(lines) > 1 else ''
                    spam_emails.append({
                        'subject': subject,
                        'body': body,
                        'label': 'spam'
                    })
        
        # Combine into DataFrame
        df = pd.DataFrame(ham_emails + spam_emails)
        print(f"Loaded {len(df)} emails from Enron dataset")
        
    except FileNotFoundError:
        # Fall back to CSV dataset
        try:
            df = pd.read_csv('data/emails.csv')
            print("Loaded emails from CSV dataset")
        except FileNotFoundError:
            print("Error: No dataset found. Please ensure either:")
            print("1. The Enron dataset is in dataset/enron/ham/ and dataset/enron/spam/")
            print("2. Or your CSV dataset is in data/emails.csv")
            return
    
    # Clean and prepare data
    df = df.dropna()  # Remove any rows with missing values
    df['label'] = df['label'].str.lower().str.strip()  # Normalize labels
    
    # Check class distribution
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    
    # Ensure we have enough samples in each class
    min_samples = 2  # Minimum samples required per class
    class_counts = df['label'].value_counts()
    if any(count < min_samples for count in class_counts):
        print("\nError: Each class must have at least 2 samples for proper training.")
        print("Current class distribution:")
        print(class_counts)
        return
    
    # Initialize detector
    detector = SpamDetector()
    
    # Extract features
    X = detector.extract_features(df)
    y = df['label']
    
    # Use a smaller test size for small datasets
    test_size = min(0.2, 1 - (4 / len(df)))  # Ensure at least 4 samples in training set
    
    # Split data using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    detector.train(X_train, y_train)
    
    # Evaluate model
    detector.evaluate(X_test, y_test)
    
    # Save the trained model
    joblib.dump(detector, 'spam_detector_model.joblib')
    print("\nModel saved to spam_detector_model.joblib")
    
    # Interactive prediction
    print("\nEnter email details to check if it's spam:")
    while True:
        print("\n" + "="*50)
        subject = input("Enter email subject (or 'quit' to exit): ")
        if subject.lower() == 'quit':
            break
            
        body = input("Enter email body: ")
        
        prediction, probability = detector.predict_single_email(subject, body)
        print("\nPrediction Results:")
        print(f"Classification: {'SPAM' if prediction == 'spam' else 'HAM'}")
        print(f"Confidence: {probability[1 if prediction == 'spam' else 0]:.2%}")
        
        # Print warning signs if spam
        if prediction == 'spam':
            print("\nWarning Signs:")
            if 'http' in body.lower():
                print("- Contains links")
            if any(word in subject.lower() for word in ['urgent', 'important', 'alert']):
                print("- Urgency indicators in subject")
            if any(word in body.lower() for word in ['offer', 'deal', 'discount', 'free']):
                print("- Contains promotional language")

if __name__ == "__main__":
    main() 