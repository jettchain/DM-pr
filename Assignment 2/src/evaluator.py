from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculates and returns a dictionary of evaluation metrics: accuracy, precision, recall, and F1 score.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    return metrics

def compare_models(results):
    """
    Compares models based on their metrics, printing a summary.
    
    Args:
        results (dict): Dictionary where keys are model names and values are metric dictionaries.
    """
    print("\nModel Comparison:")
    print("--------------------------------------------------")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
        print("--------------------------------------------------")

def aggregate_feature_importance(feature_indices, feature_scores, feature_importance_dict):
    """
    Helper function to aggregate feature importance scores across models.
    
    Args:
        feature_indices (array-like): Indices of the important features.
        feature_scores (array-like): Corresponding importance scores for the features.
        feature_importance_dict (dict): Dictionary where feature names are keys and scores are values.
    """
    for idx, score in zip(feature_indices, feature_scores):
        feature_importance_dict[idx] += score

def get_top_features_across_models(models, feature_names, top_n=10):
    """
    Aggregates and prints the top N most important features across all models for fake and genuine reviews.
    
    Args:
        models (list): List of trained models.
        feature_names (list): List of feature names.
        top_n (int): Number of top features to return for each class.
    """
    # Initialize dictionaries to accumulate importance scores
    fake_feature_importance = defaultdict(float)
    genuine_feature_importance = defaultdict(float)
    
    for model in models:
        # Check if the model is a valid object with the get_important_features method
        if not hasattr(model, 'get_important_features') or not callable(getattr(model, 'get_important_features')):
            print(f"Warning: Skipping invalid model object: {model}")
            continue

        important_features =   model.get_important_features()
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            sorted_indices = important_features.argsort()
            top_fake_indices = sorted_indices[-top_n:]
            top_genuine_indices = sorted_indices[:top_n]
            aggregate_feature_importance(top_fake_indices, important_features[top_fake_indices], fake_feature_importance)
            aggregate_feature_importance(top_genuine_indices, important_features[top_genuine_indices], genuine_feature_importance)
        else:
            # Assuming model returns a dict with fake and genuine features as indices
            top_fake_indices = important_features['fake']
            top_genuine_indices = important_features['genuine']
            aggregate_feature_importance(top_fake_indices, np.ones(len(top_fake_indices)), fake_feature_importance)
            aggregate_feature_importance(top_genuine_indices, np.ones(len(top_genuine_indices)), genuine_feature_importance)
    
    # Sort the features by aggregated importance scores
    top_fake_features = sorted(fake_feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_genuine_features = sorted(genuine_feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Print the top features across all models
    print(f"\nTop {top_n} features indicative of fake reviews across models:")
    print(", ".join([feature_names[i] for i, _ in top_fake_features]))
    
    print(f"\nTop {top_n} features indicative of genuine reviews across models:")
    print(", ".join([feature_names[i] for i, _ in top_genuine_features]))

    return {
        "fake": [feature_names[i] for i, _ in top_fake_features],
        "genuine": [feature_names[i] for i, _ in top_genuine_features]
    }
