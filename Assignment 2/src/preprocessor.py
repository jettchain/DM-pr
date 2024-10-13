import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def clean_text(text: str) -> str:
    """
    Cleans the input text by performing standard text preprocessing steps,
    including lowercasing, removing special characters, and extra whitespace.
    
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

def extract_features(data, bigrams: bool = False, use_tfidf: bool = False):
    """
    Extracts features from the text data using CountVectorizer or TfidfVectorizer,
    with options for unigrams and bigrams.
    
    Args:
        data (pd.Series): The text data for feature extraction.
        bigrams (bool): If True, extracts bigram features; otherwise, uses unigrams.
        use_tfidf (bool): If True, uses TfidfVectorizer instead of CountVectorizer.
        
    Returns:
        X (sparse matrix): Document-term matrix of extracted features.
        feature_names (list): List of feature names corresponding to the columns in X.
    """
    ngram_range = (1, 2) if bigrams else (1, 1)  # Set range based on bigrams flag
    
    # Choose between CountVectorizer and TfidfVectorizer based on use_tfidf flag
    vectorizer = TfidfVectorizer(ngram_range=ngram_range) if use_tfidf else CountVectorizer(ngram_range=ngram_range)
    
    # Fit and transform the data to produce the document-term matrix
    X = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()
    
    return X, feature_names

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
