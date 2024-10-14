# src/experiment_runner.py

from src.data_loader import load_and_split_data
from src.preprocessor import clean_text, extract_features
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.decision_tree import DecisionTreeModel
from src.models.random_forest import RandomForestModel
from src.evaluator import calculate_metrics, compare_models, get_top_features_across_models

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
    
    # Step 5: Train and evaluate models with both unigrams and bigrams
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        # Choose the appropriate feature set and ngram_type based on the model name
        if "Bigrams" in name:
            X_train = X_train_bi
            X_test = X_test_bi
            ngram_type = 'bi'
        else:
            X_train = X_train_uni
            X_test = X_test_uni
            ngram_type = 'uni'
        
        # Tune hyperparameters and train the model
        model.tune_hyperparameters(X_train, y_train, ngram_type=ngram_type)
        model.train(X_train, y_train, ngram_type=ngram_type)
        y_pred = model.predict(X_test)
        
        # Step 6: Evaluate the model
        metrics = calculate_metrics(y_test, y_pred)
        results[name] = metrics
        print(f"{name} Metrics: {metrics}")
    
    # Step 7: Compare models
    compare_models(results)

    # Step 8: Get top features
    print("Top features for unigrams:")
    get_top_features_across_models(models, feature_names_uni)
    print("\nTop features for bigrams:")
    get_top_features_across_models(models, feature_names_bi)
    
    return results
