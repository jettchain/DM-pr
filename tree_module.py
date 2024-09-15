import numpy as np
from collections import Counter
from helper_functions import best_split

class TreeNode:
    """
    A class representing a node in a decision tree.

    Attributes:
    - feature (int): Index of the feature to split on.
    - threshold (float): Threshold value for the split.
    - left (TreeNode): Left child node.
    - right (TreeNode): Right child node.
    - value (int): Class label for leaf nodes.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a TreeNode.

        Parameters:
        - feature (int): Index of the feature to split on.
        - threshold (float): Threshold value for the split.
        - left (TreeNode): Left child node.
        - right (TreeNode): Right child node.
        - value (int): Class label for leaf nodes.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def tree_grow(X, y, nmin, minleaf, nfeat):
    """
    Recursively grow a classification tree.

    Parameters:
    - X (numpy.ndarray): 2D array of attribute values (shape: [n_samples, n_features]).
    - y (numpy.ndarray): 1D array of class labels (shape: [n_samples]).
    - nmin (int): Minimum number of observations required to attempt a split.
    - minleaf (int): Minimum number of observations required in a leaf node.
    - nfeat (int): Number of features to consider when looking for the best split.

    Returns:
    - TreeNode: The root node of the decision tree.

    This function builds the decision tree by recursively finding the best split
    at each node using the Gini index, considering the minleaf and nmin constraints.
    It randomly selects nfeat features at each split to simulate the behavior
    required for random forests when nfeat < total features.
    """
    # If all labels are the same, return a leaf node with that label
    if len(np.unique(y)) == 1:
        return TreeNode(value=y[0])

    # If the number of samples is less than nmin, make this a leaf node
    if len(y) < nmin:
        majority_class = Counter(y).most_common(1)[0][0]
        return TreeNode(value=majority_class)

    # Find the best split
    split = best_split(X, y, nfeat, minleaf)

    # If no valid split is found, make this a leaf node
    if split is None:
        majority_class = Counter(y).most_common(1)[0][0]
        return TreeNode(value=majority_class)

    # Recursively build the left and right subtrees
    left_child = tree_grow(X[split["left_indices"]], y[split["left_indices"]], nmin, minleaf, nfeat)
    right_child = tree_grow(X[split["right_indices"]], y[split["right_indices"]], nmin, minleaf, nfeat)

    # Return the current node with left and right children
    return TreeNode(feature=split["feature"], threshold=split["threshold"], left=left_child, right=right_child)


def tree_pred(X, tr):
    """
    Predict class labels using a decision tree.

    Parameters:
    - X (numpy.ndarray): 2D array of attribute values (shape: [n_samples, n_features]).
    - tr (TreeNode): The root node of the decision tree.

    Returns:
    - numpy.ndarray: 1D array of predicted class labels (shape: [n_samples]).

    This function traverses the decision tree for each sample to predict its class label.
    It uses a helper function to predict the label for a single sample.
    """
    # Predict the class label for each sample in X
    y_pred = np.array([predict_single_sample(x, tr) for x in X])
    return y_pred


def predict_single_sample(x, node):
    """
    Helper function to predict the class label for a single sample.

    Parameters:
    - x (numpy.ndarray): 1D array of attribute values for a single sample.
    - node (TreeNode): The current node in the decision tree.

    Returns:
    - int: Predicted class label for the sample.

    This function recursively traverses the decision tree based on the feature values
    until it reaches a leaf node, where it returns the class label.
    """
    # If we have reached a leaf node, return its value
    if node.value is not None:
        return node.value
    # Decide whether to go left or right based on the threshold
    if x[node.feature] < node.threshold:
        return predict_single_sample(x, node.left)
    else:
        return predict_single_sample(x, node.right)
