import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

def clean_text(text: str) -> str:
    """
    Cleans the input text by performing standard text preprocessing steps.
    
    Choices made:
    1. Convert to lowercase: Standardizes all text to reduce vocabulary size.
    2. Remove special characters and numbers: Focus on alphabetic content only.
    3. Remove extra whitespace: Standardize spacing between words.
    
    We don't perform stemming or lemmatization to preserve the original form of words,
    which might be important for detecting nuances in fake vs. genuine reviews.
    
    Args:
        text (str): Raw input text to clean.
    
    Returns:
        str: Cleaned text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_features(train_texts, test_texts, ngram_range=(1, 1), use_tfidf=True):
    if use_tfidf:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    feature_names = vectorizer.get_feature_names_out()
    
    return X_train, X_test, vectorizer, feature_names

def remove_sparse_terms(X, threshold: float = 0.01):
    """
    Removes features with a frequency lower than the given threshold.
    
    Args:
        X (sparse matrix): Document-term matrix of features.
        threshold (float): Sparsity threshold; features with frequencies below this are removed.
        
    Returns:
        X_reduced (sparse matrix): Reduced feature matrix with sparse terms removed.
    """
    # Convert sparse matrix to dense and calculate the mean frequency of each feature
    mask = (X.mean(axis=0) > threshold).A1
    X_reduced = X[:, mask]
    
    return X_reduced
