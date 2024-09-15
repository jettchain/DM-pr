import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tree_module import tree_grow, tree_pred
from ensemble_module import tree_grow_b, tree_pred_b, random_forest

# Load the datasets
# Ensure the CSV files are in the same directory or provide the correct paths
train_df = pd.read_csv('eclipse-metrics-packages-2.0.csv', delimiter=';')
test_df = pd.read_csv('eclipse-metrics-packages-3.0.csv', delimiter=';')

# Select predictor variables and target variable
# Use metrics listed in Table 1 and the number of pre-release bugs (excluding AST features)
# Assuming features are in columns 4 to 44 as per assignment instructions
selected_features = ['pre'] + train_df.columns[4:44].tolist()  # Adjust indices if necessary

X_train = train_df[selected_features].values
y_train = train_df['post'].values  # Binary target variable: 0 or 1

X_test = test_df[selected_features].values
y_test = test_df['post'].values

# Ensure that class labels are binary (0 and 1)
# If necessary, map other labels to 0 and 1
# For example:
# y_train = np.where(y_train > 0, 1, 0)
# y_test = np.where(y_test > 0, 1, 0)

# Analysis Parameters
nmin = 15
minleaf = 5
total_features = X_train.shape[1]
nfeat_full = total_features  # For the full tree and bagging
nfeat_sqrt = int(np.sqrt(total_features))  # For random forest
m_trees = 100  # Number of trees for bagging and random forest

# Part 1: Train a single classification tree
print("Training a single classification tree...")
single_tree = tree_grow(X_train, y_train, nmin, minleaf, nfeat_full)
y_pred_single = tree_pred(X_test, single_tree)

# Compute metrics for the single tree
accuracy_single = accuracy_score(y_test, y_pred_single)
precision_single = precision_score(y_test, y_pred_single)
recall_single = recall_score(y_test, y_pred_single)
confusion_single = confusion_matrix(y_test, y_pred_single)

print("\nSingle Tree Results:")
print("Confusion Matrix:")
print(confusion_single)
print(f"Accuracy: {accuracy_single:.4f}")
print(f"Precision: {precision_single:.4f}")
print(f"Recall: {recall_single:.4f}")

# Part 2: Bagging with m = 100 trees
print("\nTraining an ensemble using bagging...")
bagging_trees = tree_grow_b(X_train, y_train, nmin, minleaf, nfeat_full, m_trees)
y_pred_bagging = tree_pred_b(X_test, bagging_trees)

# Compute metrics for bagging
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
precision_bagging = precision_score(y_test, y_pred_bagging)
recall_bagging = recall_score(y_test, y_pred_bagging)
confusion_bagging = confusion_matrix(y_test, y_pred_bagging)

print("\nBagging Results:")
print("Confusion Matrix:")
print(confusion_bagging)
print(f"Accuracy: {accuracy_bagging:.4f}")
print(f"Precision: {precision_bagging:.4f}")
print(f"Recall: {recall_bagging:.4f}")

# Part 3: Random Forests with nfeat = sqrt(total_features)
print("\nTraining a random forest...")
random_forest_trees = random_forest(X_train, y_train, nmin, minleaf, nfeat_sqrt, m_trees)
y_pred_rf = tree_pred_b(X_test, random_forest_trees)

# Compute metrics for random forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
confusion_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("Confusion Matrix:")
print(confusion_rf)
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")

# Additional: Print the first three splits of the single tree
def print_first_three_splits(node, depth=0):
    if node is None or node.value is not None or depth > 1:
        return
    print(f"Depth {depth}: Split on feature {node.feature} with threshold {node.threshold}")
    print(f"Left child at depth {depth + 1}:")
    if node.left.value is not None:
        print(f"  Leaf node with class {node.left.value}")
    else:
        print(f"  Split on feature {node.left.feature} with threshold {node.left.threshold}")
    print(f"Right child at depth {depth + 1}:")
    if node.right.value is not None:
        print(f"  Leaf node with class {node.right.value}")
    else:
        print(f"  Split on feature {node.right.feature} with threshold {node.right.threshold}")

    # Recurse for next level
    print_first_three_splits(node.left, depth + 1)
    print_first_three_splits(node.right, depth + 1)

print("\nFirst Three Splits of the Single Tree:")
print_first_three_splits(single_tree)

# Additional: Perform statistical test to compare accuracies
from statsmodels.stats.contingency_tables import mcnemar

# Function to perform McNemar's test between two models
def compare_models(y_true, y_pred1, y_pred2):
    contingency_table = np.zeros((2, 2))
    for i in range(len(y_true)):
        if y_pred1[i] == y_true[i] and y_pred2[i] == y_true[i]:
            continue  # Both correct
        elif y_pred1[i] == y_true[i]:
            contingency_table[0, 1] += 1  # Model 1 correct, Model 2 incorrect
        elif y_pred2[i] == y_true[i]:
            contingency_table[1, 0] += 1  # Model 1 incorrect, Model 2 correct
        else:
            continue  # Both incorrect
    result = mcnemar(contingency_table, exact=True)
    return result.pvalue

# Compare Single Tree and Bagging
p_value_single_bagging = compare_models(y_test, y_pred_single, y_pred_bagging)
print(f"\nMcNemar's Test between Single Tree and Bagging p-value: {p_value_single_bagging:.4f}")

# Compare Bagging and Random Forest
p_value_bagging_rf = compare_models(y_test, y_pred_bagging, y_pred_rf)
print(f"McNemar's Test between Bagging and Random Forest p-value: {p_value_bagging_rf:.4f}")

# Compare Single Tree and Random Forest
p_value_single_rf = compare_models(y_test, y_pred_single, y_pred_rf)
print(f"McNemar's Test between Single Tree and Random Forest p-value: {p_value_single_rf:.4f}")
