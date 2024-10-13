from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
