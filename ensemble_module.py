import numpy as np
from collections import Counter
from tree_module import tree_grow, tree_pred

def tree_grow_b(X, y, nmin, minleaf, nfeat, m):
    """
    Grow multiple trees using bootstrap samples (Bagging).

    Parameters:
    - X (numpy.ndarray): 2D array of attribute values.
    - y (numpy.ndarray): 1D array of class labels.
    - nmin (int): Minimum number of observations required to attempt a split.
    - minleaf (int): Minimum number of observations required in a leaf node.
    - nfeat (int): Number of features to consider when looking for the best split.
    - m (int): Number of bootstrap samples (number of trees in the ensemble).

    Returns:
    - list: A list of TreeNode objects representing the ensemble of decision trees.

    This function implements Bagging by training 'm' decision trees on 'm' bootstrap
    samples drawn from the original dataset. Each tree is grown using the 'tree_grow' function.
    """
    n_samples = X.shape[0]
    trees = []

    for i in range(m):
        # Generate bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        # Grow a tree on the bootstrap sample
        tree = tree_grow(X_sample, y_sample, nmin, minleaf, nfeat)
        trees.append(tree)

    return trees


def tree_pred_b(X, trees):
    """
    Predict class labels using an ensemble of trees.

    Parameters:
    - X (numpy.ndarray): 2D array of attribute values.
    - trees (list): A list of TreeNode objects representing the ensemble.

    Returns:
    - numpy.ndarray: 1D array of predicted class labels.

    This function predicts class labels by aggregating the predictions from each tree
    and using majority voting for the final prediction.
    """
    # Collect predictions from all trees
    predictions = np.array([tree_pred(X, tree) for tree in trees])

    # Perform majority voting
    y_pred = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions)

    return y_pred


def random_forest(X, y, nmin, minleaf, nfeat, m):
    """
    Implement a random forest classifier.

    Parameters:
    - X (numpy.ndarray): 2D array of attribute values.
    - y (numpy.ndarray): 1D array of class labels.
    - nmin (int): Minimum number of observations required to attempt a split.
    - minleaf (int): Minimum number of observations required in a leaf node.
    - nfeat (int): Number of features to consider when looking for the best split (typically sqrt of total features).
    - m (int): Number of trees in the forest.

    Returns:
    - list: A list of TreeNode objects representing the random forest.

    The random forest is built by growing multiple trees using bootstrap samples
    and considering only 'nfeat' random features at each split, which introduces
    randomness and reduces correlation between trees.
    """
    # The implementation is the same as tree_grow_b but with a smaller nfeat
    return tree_grow_b(X, y, nmin, minleaf, nfeat, m)
