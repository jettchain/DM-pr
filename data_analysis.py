import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tree_module import tree_grow, tree_pred
from ensemble_module import tree_grow_b, tree_pred_b, random_forest

# Load the datasets
# Replace 'path_to_train_csv' and 'path_to_test_csv' with actual file paths
train_df = pd.read_csv('eclipse-metrics-packages-2.0.csv', delimiter=';')
test_df = pd.read_csv('eclipse-metrics-packages-3.0.csv', delimiter=';')

# Select predictor variables and target variable
# According to the assignment, use metrics from Table 1 and the number of pre-release bugs
# Exclude features derived from the abstract syntax tree
selected_features = ['pre'] + train_df.columns[4:44].tolist()  # Adjust indices as needed

X_train = train_df[selected_features].values
y_train = train_df['post'].values  # Binary target variable: 0 or 1

X_test = test_df[selected_features].values
y_test = test_df['post'].values

# Ensure that class labels are binary (0 and 1)
# If necessary, map other labels to 0 and 1

# Analysis Parameters
nmin = 15
minleaf = 5
total_features = X_train.shape[1]
nfeat_full = total_features  # For the full tree and bagging
nfeat_sqrt = int(np.sqrt(total_features))  # For random forest
m_trees = 100  # Number of trees for bagging and random forest

# 1. Train a single classification tree
single_tree = tree_grow(X_train, y_train, nmin, minleaf, nfeat_full)
y_pred_single = tree_pred(X_test, single_tree)

# Compute metrics for the single tree
accuracy_single = accuracy_score(y_test, y_pred_single)
precision_single = precision_score(y_test, y_pred_single)
recall_single = recall_score(y_test, y_pred_single)
confusion_single = confusion_matrix(y_test, y_pred_single)

print("Single Tree Results:")
print("Confusion Matrix:")
print(confusion_single)
print(f"Accuracy: {accuracy_single:.4f}")
print(f"Precision: {precision_single:.4f}")
print(f"Recall: {recall_single:.4f}")

# 2. Bagging with m = 100 trees
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

# 3. Random Forests with nfeat = sqrt(total_features)
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

# Additional analysis and statistical tests would be performed here.
