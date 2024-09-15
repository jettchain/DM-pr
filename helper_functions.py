import numpy as np
from collections import Counter

def gini(y):
    """
    Calculate the Gini impurity for a set of labels.

    Parameters:
    - y (numpy.ndarray): 1D array of class labels.

    Returns:
    - float: Gini impurity score.

    This function computes the Gini impurity, which measures the likelihood
    of an incorrect classification of a new instance if it was randomly labeled
    according to the distribution of labels in the subset.
    """
    m = len(y)
    if m == 0:
        return 0.0
    counts = np.bincount(y)
    probabilities = counts / m
    return 1.0 - np.sum(probabilities ** 2)


def best_split(X, y, nfeat, minleaf):
    """
    Find the best feature and threshold to split on.

    Parameters:
    - X (numpy.ndarray): 2D array of attribute values.
    - y (numpy.ndarray): 1D array of class labels.
    - nfeat (int): Number of features to consider for the split.
    - minleaf (int): Minimum number of observations required in a leaf node.

    Returns:
    - dict or None: Dictionary containing the best split information or None if no valid split is found.

    The function evaluates all possible splits on randomly selected features
    (up to nfeat) and selects the split that results in the lowest weighted Gini impurity.
    It ensures that each resulting node satisfies the minleaf constraint.
    """
    m, n_features = X.shape
    if m <= 1:
        return None

    # Randomly select nfeat features without replacement
    features = np.random.choice(n_features, size=min(nfeat, n_features), replace=False)

    best_gini = np.inf
    best_split = None

    # Iterate over selected features
    for feature in features:
        # Sort data along the feature
        X_feature = X[:, feature]
        sorted_indices = np.argsort(X_feature)
        X_sorted, y_sorted = X_feature[sorted_indices], y[sorted_indices]

        # Identify potential split positions
        for i in range(minleaf, m - minleaf + 1):
            if X_sorted[i - 1] == X_sorted[i]:
                continue  # Skip if the threshold would not separate data

            threshold = (X_sorted[i - 1] + X_sorted[i]) / 2

            # Split data
            left_indices = sorted_indices[:i]
            right_indices = sorted_indices[i:]

            # Compute Gini impurity for the split
            left_gini = gini(y[left_indices])
            right_gini = gini(y[right_indices])

            # Weighted average of Gini impurity
            total_gini = (len(left_indices) * left_gini + len(right_indices) * right_gini) / m

            # Update best split if this split has a lower Gini impurity
            if total_gini < best_gini:
                best_gini = total_gini
                best_split = {
                    "feature": feature,
                    "threshold": threshold,
                    "left_indices": left_indices,
                    "right_indices": right_indices
                }

    return best_split
