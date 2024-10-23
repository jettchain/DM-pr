from src.data_loader import load_and_split_data
from src.preprocessor import clean_text, extract_features
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.decision_tree import DecisionTreeModel
from src.models.random_forest import RandomForestModel
from src.evaluator import (
    calculate_metrics, compare_models, get_top_features_across_models,
    compare_accuracies, answer_questions, visualize_results
)
import os

def run_experiment(root_dir: str):
    """
    Runs the experiment workflow: loading data, preprocessing, training, and evaluation.
    Uses folds 1-4 for training and fold 5 for final testing.
    
    Args:
        root_dir (str): The root directory where the data is stored.
    """
    # Step 1: Load and split the data
    train_data, test_data = load_and_split_data(root_dir)
    
    # Step 2: Preprocess the text data
    train_data['text'] = train_data['text'].apply(clean_text)
    test_data['text'] = test_data['text'].apply(clean_text)

    feature_configs = [
        ("Unigrams", (1, 1)),
        ("Bigrams", (2, 2)),
        ("UniBigrams", (1, 2))
    ]

    vectorizer_types = [
        ("Count", False),
        ("TfIdf", True)
    ]

    X_train_features = {}
    X_test_features = {}
    feature_names = {}

    for feat_name, ngram_range in feature_configs:
        for vec_name, use_tfidf in vectorizer_types:
            key = f"{feat_name}_{vec_name}"
            X_train, X_test, _, feat_names = extract_features(
                train_data['text'], test_data['text'], 
                ngram_range=ngram_range, use_tfidf=use_tfidf
            )
            X_train_features[key] = X_train
            X_test_features[key] = X_test
            feature_names[key] = feat_names

    y_train = train_data['label']
    y_test = test_data['label']

    models = {
        "Naive Bayes": NaiveBayesClassifier,
        "Logistic Regression": LogisticRegressionClassifier,
        "Decision Tree": DecisionTreeModel,
        "Random Forest": RandomForestModel
    }

    results = {}
    trained_models = {}

    for model_name, model_class in models.items():
        for feat_config, _ in feature_configs:
            for vec_type, _ in vectorizer_types:
                key = f"{feat_config}_{vec_type}"
                full_name = f"{model_name} {key}"
                print(f"Training {full_name}...")
                
                model = model_class()
                X_train = X_train_features[key]
                X_test = X_test_features[key]
                
                model.tune_hyperparameters(X_train, y_train, ngram_type=key)
                model.train(X_train, y_train, ngram_type=key, use_feature_selection=True)
                y_pred = model.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred)
                results[full_name] = metrics
                trained_models[full_name] = model
                print(f"{full_name} Metrics: {metrics}")

    results_dir = os.path.join("Assignment 2", "results")
    os.makedirs(results_dir, exist_ok=True)

    compare_models(results, save_dir=results_dir)
    visualize_results(results, trained_models, X_test_features, y_test, results_dir)

    for feat_config, _ in feature_configs:
        for vec_type, _ in vectorizer_types:
            key = f"{feat_config}_{vec_type}"
            relevant_models = {k: v for k, v in trained_models.items() if key in k}
            top_features = get_top_features_across_models(relevant_models, feature_names[key], save_dir=results_dir, prefix=key)
    
    compare_accuracies(results, save_dir=results_dir)
    answer_questions(results, feature_names, save_dir=results_dir)
    
    return results

if __name__ == "__main__":
    root_dir = "Assignment 2/op_spam_v1.4/op_spam_v1.4/"
    run_experiment(root_dir)
