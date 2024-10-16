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

    # Step 3: Extract features (try both unigrams and bigrams)
    X_train_uni, X_test_uni, vectorizer_uni, feature_names_uni = extract_features(train_data['text'], test_data['text'], bigrams=False, use_tfidf=True)
    X_train_bi, X_test_bi, vectorizer_bi, feature_names_bi = extract_features(train_data['text'], test_data['text'], bigrams=True, use_tfidf=True)
    y_train = train_data['label']
    y_test = test_data['label']

    # Step 4: Initialize models
    models = {
        "Naive Bayes Unigrams": NaiveBayesClassifier(),
        "Naive Bayes Bigrams": NaiveBayesClassifier(),
        "Logistic Regression Unigrams": LogisticRegressionClassifier(),
        "Logistic Regression Bigrams": LogisticRegressionClassifier(),
        "Decision Tree Unigrams": DecisionTreeModel(),
        "Decision Tree Bigrams": DecisionTreeModel(),
        "Random Forest Unigrams": RandomForestModel(),
        "Random Forest Bigrams": RandomForestModel()
    }
    
    # Step 5: Train models and collect results
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        if "Bigrams" in name:
            X_train, X_test = X_train_bi, X_test_bi
            ngram_type = 'bi'
        else:
            X_train, X_test = X_train_uni, X_test_uni
            ngram_type = 'uni'
        
        model.tune_hyperparameters(X_train, y_train, ngram_type=ngram_type)
        model.train(X_train, y_train, ngram_type=ngram_type, use_feature_selection=True)
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        results[name] = metrics
        print(f"{name} Metrics: {metrics}")

    # Step 6: Evaluate and visualize results
    results_dir = os.path.join("Assignment 2", "results")
    os.makedirs(results_dir, exist_ok=True)

    compare_models(results, save_dir=results_dir)
    visualize_results(results, models, X_test_uni, X_test_bi, y_test, results_dir)

    uni_top_features = get_top_features_across_models({k: v for k, v in models.items() if "Unigrams" in k}, feature_names_uni, save_dir=results_dir)
    bi_top_features = get_top_features_across_models({k: v for k, v in models.items() if "Bigrams" in k}, feature_names_bi, save_dir=results_dir)
    
    compare_accuracies(results, save_dir=results_dir)
    answer_questions(results, uni_top_features, bi_top_features, save_dir=results_dir)
    
    return results

if __name__ == "__main__":
    root_dir = "Assignment 2/op_spam_v1.4/op_spam_v1.4/"
    run_experiment(root_dir)
