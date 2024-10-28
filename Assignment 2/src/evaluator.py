from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from io import StringIO
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import os
import json


# evaluator.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def compare_models(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'model_comparison.txt'), 'w') as f:
        f.write("Model Comparison:\n")
        f.write("--------------------------------------------------\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            for metric, score in metrics.items():
                if isinstance(score, (int, float)):
                    f.write(f"  {metric}: {score:.4f}\n")
            f.write("--------------------------------------------------\n")
    print(f"Model comparison results have been saved to {save_dir}")

def get_top_features_across_models(models, feature_names, save_dir, top_n=5, prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    all_top_features = {}
    for model_name, model in models.items():
        if hasattr(model, 'get_important_features'):
            important_features = model.get_important_features()
            deceptive_indices = important_features.get('Deceptive', [])
            truthful_indices = important_features.get('Truthful', [])
            deceptive_features = [feature_names[idx] for idx in deceptive_indices if idx < len(feature_names)]
            truthful_features = [feature_names[idx] for idx in truthful_indices if idx < len(feature_names)]
            all_top_features[model_name] = {
                "Deceptive": deceptive_features[:top_n],
                "Truthful": truthful_features[:top_n]
            }
    with open(os.path.join(save_dir, f'top_features_{prefix}.json'), 'w') as f:
        json.dump(all_top_features, f, indent=2)
    print(f"Top features for {prefix} have been saved to {save_dir}")
    return all_top_features

def compare_accuracies(models, save_dir):
    import numpy as np
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import multipletests
    import os
    import json

    os.makedirs(save_dir, exist_ok=True)
    model_names = list(models.keys())
    comparisons = []
    p_values = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1_name = model_names[i]
            model2_name = model_names[j]

            # Ensure model pairs belong to the same category
            if any(x in model1_name and x in model2_name for x in ["Unigrams_Count", "Unigrams_TfIdf", "Bigrams_Count", "Bigrams_TfIdf", "UniBigrams_Count", "UniBigrams_TfIdf"]):
                acc1 = models[model1_name].accuracy_scores
                acc2 = models[model2_name].accuracy_scores

                # Perform paired t-test if both models have scores
                if len(acc1) > 0 and len(acc2) > 0:
                    stat, p_value = ttest_rel(acc1, acc2)
                    mean_diff = np.mean(acc1) - np.mean(acc2)
                    significant = bool(p_value < 0.05)  # Convert to Python bool

                    comparisons.append({
                        "models": f"{model1_name} vs {model2_name}",
                        "mean_accuracy_difference": mean_diff,
                        "p_value": float(p_value),  # Ensure p_value is float
                        "significant": significant
                    })
                    p_values.append(p_value)

    # Multiple testing correction
    corrected_p_values, corrected_significant = multipletests(p_values, alpha=0.05, method='bonferroni')[:2]
    corrected_significant = [bool(x) for x in corrected_significant]  # Convert to Python bool

    for comparison, corrected_p, corrected_sig in zip(comparisons, corrected_p_values, corrected_significant):
        comparison["corrected_p_value"] = float(corrected_p)  # Ensure corrected_p_value is float
        comparison["corrected_significant"] = corrected_sig

    # Explicitly convert any remaining numpy types to JSON-compatible types
    comparisons = [
        {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in comp.items()} for comp in comparisons
    ]

    # Save results to JSON
    with open(os.path.join(save_dir, 'accuracy_comparison_ttest.json'), 'w') as f:
        json.dump(comparisons, f, indent=2)


def answer_questions(results, feature_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    answers = {}
    nb_uni_acc = results["Naive Bayes Unigrams_Count"]["accuracy"]
    lr_uni_acc = results["Logistic Regression Unigrams_Count"]["accuracy"]
    better_model = 'Naive Bayes' if nb_uni_acc > lr_uni_acc else 'Logistic Regression'
    answers["q1"] = {
        "nb_accuracy": float(nb_uni_acc),
        "lr_accuracy": float(lr_uni_acc),
        "better_model": better_model
    }
    rf_uni_acc = results["Random Forest Unigrams_Count"]["accuracy"]
    best_linear_acc = max(nb_uni_acc, lr_uni_acc)
    improvement = rf_uni_acc > best_linear_acc
    answers["q2"] = {
        "rf_accuracy": float(rf_uni_acc),
        "best_linear_accuracy": float(best_linear_acc),
        "improvement": bool(improvement)
    }
    with open(os.path.join(save_dir, 'experiment_answers.json'), 'w') as f:
        json.dump(answers, f, indent=2)
    print(f"Experiment answers have been saved to {save_dir}")
    return answers


def visualize_results(results, models, X_test_features, y_test, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    # Accuracy Comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))
    plt.close()
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        feature_key = '_'.join(name.split()[1:])
        X_test = X_test_features.get(feature_key)
        if X_test is not None and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()
    # Confusion Matrices
    num_models = len(models)
    cols = 3
    rows = (num_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for i, (name, model) in enumerate(models.items()):
        feature_key = '_'.join(name.split()[1:])
        X_test = X_test_features.get(feature_key)
        if X_test is not None:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'))
    plt.close()
