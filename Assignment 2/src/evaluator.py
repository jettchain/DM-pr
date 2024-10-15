from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

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
        "fake": [(feature_names[i], score) for i, score in top_fake_features if i < len(feature_names)][:top_n],
        "genuine": [(feature_names[i], score) for i, score in top_genuine_features if i < len(feature_names)][:top_n]
    }

def compare_accuracies(results):
    # Focus on test set results
    model_names = [name for name in results.keys() if 'Test' in name]
    accuracies = [results[model]['accuracy'] for model in model_names]
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            acc1, acc2 = accuracies[i], accuracies[j]
            
            # Perform Wilcoxon signed-rank test
            _, p_value = wilcoxon([acc1], [acc2])
            
            print(f"{model1} vs {model2}:")
            print(f"  Accuracy difference: {abs(acc1 - acc2):.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'}\n")

def answer_questions(results, uni_top_features, bi_top_features):
    print("\nAnswering experiment questions:")
    
    # Question 1
    print("\n1. Comparison of Naive Bayes and Logistic Regression:")
    nb_uni_acc = results["Naive Bayes Unigrams Test"]["accuracy"]
    lr_uni_acc = results["Logistic Regression Unigrams Test"]["accuracy"]
    print(f"  Naive Bayes (unigrams) accuracy: {nb_uni_acc:.4f}")
    print(f"  Logistic Regression (unigrams) accuracy: {lr_uni_acc:.4f}")
    print(f"  {'Naive Bayes' if nb_uni_acc > lr_uni_acc else 'Logistic Regression'} performs better.")
    print("  This comparison shows how a generative model (Naive Bayes) compares to a discriminative model (Logistic Regression) for this task.")

    # Question 2
    print("\n2. Performance of Random Forest compared to linear classifiers:")
    rf_uni_acc = results["Random Forest Unigrams Test"]["accuracy"]
    best_linear_acc = max(nb_uni_acc, lr_uni_acc)
    print(f"  Random Forest (unigrams) accuracy: {rf_uni_acc:.4f}")
    print(f"  Best linear classifier accuracy: {best_linear_acc:.4f}")
    print(f"  Random Forest {'improves' if rf_uni_acc > best_linear_acc else 'does not improve'} on the performance of linear classifiers.")
    print("  This shows whether the ensemble of non-linear classifiers (Random Forest) can capture more complex patterns than linear models.")

    # Question 3
    print("\n3. Impact of adding bigram features:")
    for model_type in ["Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest"]:
        uni_acc = results[f"{model_type} Unigrams Test"]["accuracy"]
        bi_acc = results[f"{model_type} Bigrams Test"]["accuracy"]
        print(f"  {model_type}:")
        print(f"    Unigrams accuracy: {uni_acc:.4f}")
        print(f"    Bigrams accuracy: {bi_acc:.4f}")
        print(f"    Performance {'improves' if bi_acc > uni_acc else 'does not improve'} with bigrams.")
    print("  This comparison shows whether capturing word pairs (bigrams) provides additional useful information for detecting fake reviews.")

    # Questions 4 and 5
    print("\n4. Five most important terms pointing towards a fake review:")
    print("  Unigrams:", ", ".join([f[0] for f in uni_top_features["fake"][:5]]))
    print("  Bigrams:", ", ".join([f[0] for f in bi_top_features["fake"][:5]]))
    print("  These terms are most indicative of deceptive reviews across all models.")

    print("\n5. Five most important terms pointing towards a genuine review:")
    print("  Unigrams:", ", ".join([f[0] for f in uni_top_features["genuine"][:5]]))
    print("  Bigrams:", ", ".join([f[0] for f in bi_top_features["genuine"][:5]]))
    print("  These terms are most indicative of truthful reviews across all models.")

def visualize_results(results, models, X_test, y_test):
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(results.keys()), y=[r['accuracy'] for r in results.values()])
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()

    # ROC curves
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if 'Unigrams' in name:  # Plot only unigram models for clarity
            y_pred_proba = model.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()
    for i, (name, model) in enumerate(models.items()):
        if 'Unigrams' in name:  # Plot only unigram models for clarity
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()