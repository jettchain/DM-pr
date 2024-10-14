from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import warnings

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

def get_top_features_across_models(models, feature_names, top_n=5):
    fake_feature_importance = defaultdict(float)
    genuine_feature_importance = defaultdict(float)
    
    for model_name, model in models.items():
        if not hasattr(model, 'get_important_features') or not callable(getattr(model, 'get_important_features')):
            print(f"Warning: Skipping invalid model object: {model_name}")
            continue

        important_features = model.get_important_features()
        
        if isinstance(important_features, dict):
            for idx in important_features['Deceptive']:
                fake_feature_importance[idx] += 1
            for idx in important_features['Truthful']:
                genuine_feature_importance[idx] += 1
        else:
            sorted_indices = important_features.argsort()
            top_fake_indices = sorted_indices[-top_n:]
            top_genuine_indices = sorted_indices[:top_n]
            for idx in top_fake_indices:
                fake_feature_importance[idx] += important_features[idx]
            for idx in top_genuine_indices:
                genuine_feature_importance[idx] += important_features[idx]
    
    top_fake_features = sorted(fake_feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_genuine_features = sorted(genuine_feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    fake_features = [feature_names[i] for i, _ in top_fake_features if i < len(feature_names)]
    genuine_features = [feature_names[i] for i, _ in top_genuine_features if i < len(feature_names)]

    # Ensure we always have exactly top_n features
    while len(fake_features) < top_n:
        fake_features.append("N/A")
    while len(genuine_features) < top_n:
        genuine_features.append("N/A")

    print(f"\nTop {top_n} features indicative of fake reviews across models:")
    print(", ".join(fake_features))
    
    print(f"\nTop {top_n} features indicative of genuine reviews across models:")
    print(", ".join(genuine_features))

    return {
        "fake": fake_features[:top_n],
        "genuine": genuine_features[:top_n]
    }
