from sklearn.feature_selection import SelectKBest, chi2

def select_features_naive_bayes(X, y, num_features: int = 1000):
    """
    Selects the top features based on chi-square statistic, specifically for Naive Bayes.
    
    Args:
        X (sparse matrix): Feature matrix (e.g., from CountVectorizer).
        y (array-like): Target labels.
        num_features (int): Number of top features to select.
        
    Returns:
        X_selected (sparse matrix): Reduced feature matrix with the top selected features.
        selected_feature_indices (list): Indices of the selected features.
    """
    # Use chi-square to select the most relevant features
    selector = SelectKBest(score_func=chi2, k=num_features)
    X_selected = selector.fit_transform(X, y)
    
    # Get indices of selected features for potential inspection
    selected_feature_indices = selector.get_support(indices=True)
    
    return X_selected, selected_feature_indices
