from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys
from io import StringIO

class PrintCapture:
    def __init__(self):
        self.old_stdout = sys.stdout
        self.output = StringIO()

    def __enter__(self):
        sys.stdout = self.output
        return self

    def __exit__(self, *args):
        sys.stdout = self.old_stdout

    def get_output(self):
        return self.output.getvalue()

def save_output(output, save_dir, filename):
    with open(os.path.join(save_dir, filename), 'w') as f:
        f.write(output)

def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    return metrics

def compare_models(results, save_dir):
    with PrintCapture() as capture:
        print("\nModel Comparison:")
        print("--------------------------------------------------")
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            for metric, score in metrics.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.4f}")
            print("--------------------------------------------------")
    
    output = capture.get_output()
    save_output(output, save_dir, 'model_comparison.txt')
    print(f"Model comparison results have been saved to {save_dir}")

def get_top_features_across_models(models, feature_names, save_dir, top_n=5, prefix=""):
    with PrintCapture() as capture:
        fake_feature_importance = defaultdict(float)
        genuine_feature_importance = defaultdict(float)
        
        for model_name, model in models.items():
            if not hasattr(model, 'get_important_features') or not callable(getattr(model, 'get_important_features')):
                print(f"Warning: Skipping invalid model object: {model_name}")
                continue

            important_features = model.get_important_features()
            
            if isinstance(important_features, dict):
                for idx in important_features['Deceptive']:
                    fake_feature_importance[feature_names[idx]] += 1
                for idx in important_features['Truthful']:
                    genuine_feature_importance[feature_names[idx]] += 1
            else:
                sorted_indices = important_features.argsort()
                top_fake_indices = sorted_indices[-top_n:]
                top_genuine_indices = sorted_indices[:top_n]
                for idx in top_fake_indices:
                    fake_feature_importance[feature_names[idx]] += important_features[idx]
                for idx in top_genuine_indices:
                    genuine_feature_importance[feature_names[idx]] += important_features[idx]
        
        top_fake_features = sorted(fake_feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_genuine_features = sorted(genuine_feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fake_features = [f[0] for f in top_fake_features]
        genuine_features = [f[0] for f in top_genuine_features]

        print(f"\nTop {top_n} features indicative of fake reviews across models:")
        print(", ".join(fake_features))
        
        print(f"\nTop {top_n} features indicative of genuine reviews across models:")
        print(", ".join(genuine_features))

    output = capture.get_output()
    output_filename = f'top_features_{prefix}.txt'
    json_filename = f'top_features_{prefix}.json'
    save_output(output, save_dir, output_filename)
    
    top_features = {
        "fake": top_fake_features,
        "genuine": top_genuine_features
    }
    
    with open(os.path.join(save_dir, json_filename), 'w') as f:
        json.dump(top_features, f, indent=2)
    
    print(f"Top features for {prefix} have been saved to {save_dir}")
    return top_features

def compare_accuracies(results, save_dir):
    with PrintCapture() as capture:
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in model_names]
        
        comparisons = []
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                acc1, acc2 = accuracies[i], accuracies[j]
                
                _, p_value = wilcoxon([acc1], [acc2])
                
                comparison = {
                    "models": f"{model1} vs {model2}",
                    "accuracy_difference": abs(acc1 - acc2),
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
                comparisons.append(comparison)
                
                print(f"{model1} vs {model2}:")
                print(f"  Accuracy difference: {abs(acc1 - acc2):.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'}\n")
    
    output = capture.get_output()
    save_output(output, save_dir, 'accuracy_comparison.txt')
    
    with open(os.path.join(save_dir, 'accuracy_comparison.json'), 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    print(f"Accuracy comparison results have been saved to {save_dir}")
    return comparisons

def answer_questions(results, feature_names, save_dir):
    with PrintCapture() as capture:
        print("\nAnswering experiment questions:")
        
        answers = {}
        
        print("\n1. Comparison of Naive Bayes and Logistic Regression:")
        nb_uni_acc = results["Naive Bayes Unigrams_Count"]["accuracy"]
        lr_uni_acc = results["Logistic Regression Unigrams_Count"]["accuracy"]
        better_model = 'Naive Bayes' if nb_uni_acc > lr_uni_acc else 'Logistic Regression'
        print(f"  Naive Bayes (unigrams) accuracy: {nb_uni_acc:.4f}")
        print(f"  Logistic Regression (unigrams) accuracy: {lr_uni_acc:.4f}")
        print(f"  {better_model} performs better.")
        print("  This comparison shows how a generative model (Naive Bayes) compares to a discriminative model (Logistic Regression) for this task.")
        answers["q1"] = {
            "nb_accuracy": nb_uni_acc,
            "lr_accuracy": lr_uni_acc,
            "better_model": better_model
        }

        print("\n2. Performance of Random Forest compared to linear classifiers:")
        rf_uni_acc = results["Random Forest Unigrams_Count"]["accuracy"]
        best_linear_acc = max(nb_uni_acc, lr_uni_acc)
        print(f"  Random Forest (unigrams) accuracy: {rf_uni_acc:.4f}")
        print(f"  Best linear classifier accuracy: {best_linear_acc:.4f}")
        print(f"  Random Forest {'improves' if rf_uni_acc > best_linear_acc else 'does not improve'} on the performance of linear classifiers.")
        print("  This shows whether the ensemble of non-linear classifiers (Random Forest) can capture more complex patterns than linear models.")

        print("\n3. Impact of adding bigram features:")
        for model_type in ["Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest"]:
            uni_acc = results[f"{model_type} Unigrams_Count"]["accuracy"]
            bi_acc = results[f"{model_type} Bigrams_Count"]["accuracy"]
            print(f"  {model_type}:")
            print(f"    Unigrams accuracy: {uni_acc:.4f}")
            print(f"    Bigrams accuracy: {bi_acc:.4f}")
            print(f"    Performance {'improves' if bi_acc > uni_acc else 'does not improve'} with bigrams.")
        print("  This comparison shows whether capturing word pairs (bigrams) provides additional useful information for detecting fake reviews.")

        print("\n4. Five most important terms pointing towards a fake review:")
        uni_top_features = get_top_features_across_models({k: v for k, v in results.items() if "Unigrams_Count" in k}, feature_names["Unigrams_Count"], save_dir, prefix="Unigrams_Count")
        bi_top_features = get_top_features_across_models({k: v for k, v in results.items() if "Bigrams_Count" in k}, feature_names["Bigrams_Count"], save_dir, prefix="Bigrams_Count")
        print("  Unigrams:", ", ".join([f[0] for f in uni_top_features["fake"][:5]]))
        print("  Bigrams:", ", ".join([f[0] for f in bi_top_features["fake"][:5]]))
        print("  These terms are most indicative of deceptive reviews across all models.")

        print("\n5. Five most important terms pointing towards a genuine review:")
        print("  Unigrams:", ", ".join([f[0] for f in uni_top_features["genuine"][:5]]))
        print("  Bigrams:", ", ".join([f[0] for f in bi_top_features["genuine"][:5]]))
        print("  These terms are most indicative of truthful reviews across all models.")

    output = capture.get_output()
    save_output(output, save_dir, 'experiment_answers.txt')
    
    with open(os.path.join(save_dir, 'experiment_answers.json'), 'w') as f:
        json.dump(answers, f, indent=2)
    
    print(f"Experiment answers have been saved to {save_dir}")
    return answers

def visualize_results(results, models, X_test_features, y_test, results_dir):
    plt.figure(figsize=(20, 10))
    sns.barplot(x=list(results.keys()), y=[r['accuracy'] for r in results.values()])
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))
    plt.close()

    plt.figure(figsize=(20, 15))
    for name, model in models.items():
        feature_key = '_'.join(name.split()[1:])  # Extract feature configuration from model name
        X_test = X_test_features[feature_key]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()

    fig, axes = plt.subplots(4, 6, figsize=(30, 20))
    axes = axes.ravel()
    for i, (name, model) in enumerate(models.items()):
        feature_key = '_'.join(name.split()[1:])
        X_test = X_test_features[feature_key]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'))
    plt.close()
