from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from .base_model import BaseModel
import os
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
import ast 
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score
import ast



class NaiveBayesClassifier(BaseModel):

    def __init__(self):
        self.model = MultinomialNB()
        self.feature_selector = None
        self.num_features = None
        self.accuracy_scores = []
        self.hyperparams = None
        self.model_filename = None
        self.feature_selector_filename = None

    def train(self, X_train, y_train, X_test, y_test, ngram_type='uni', num_runs=50, override=False, use_feature_selection=True):
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"naive_bayes_model_{ngram_type}.pkl")
        self.feature_selector_filename = os.path.join(weights_dir, f"naive_bayes_feature_selector_{ngram_type}.pkl")
        hyperparams_filename = os.path.join(weights_dir, f"naive_bayes_params_{ngram_type}.pkl")

        # Load hyperparameters if available
        if self.hyperparams is None:
            if os.path.exists(hyperparams_filename):
                loaded_data = joblib.load(hyperparams_filename)
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    best_params, self.num_features = loaded_data

                    # Ensure best_params is a dictionary. If it's a string, convert it to a dictionary
                    if isinstance(best_params, str):
                        try:
                            best_params = ast.literal_eval(best_params)  # Safely convert string to dictionary
                        except (SyntaxError, ValueError) as e:
                            raise ValueError(f"Failed to parse best_params: {e}")

                    # Validate that best_params is a dictionary
                    if not isinstance(best_params, dict):
                        raise ValueError("Loaded best_params must be a dictionary.")

                    self.hyperparams = best_params
                else:
                    raise ValueError("Hyperparameters not found in the correct format. Please run tune_hyperparameters before training.")
            else:
                raise ValueError("Hyperparameters not found. Please run tune_hyperparameters before training.")

        # Load the entire class instance if available
        if os.path.exists(self.model_filename) and not override:
            # Load the entire class instance
            loaded_self = joblib.load(self.model_filename)
            self.__dict__.update(loaded_self.__dict__)
        else:
            self.accuracy_scores = []
            if use_feature_selection and self.num_features is not None:
                self.feature_selector = SelectKBest(chi2, k=self.num_features)
                X_train_fs = self.feature_selector.fit_transform(X_train, y_train)
                X_test_fs = self.feature_selector.transform(X_test)
                joblib.dump(self.feature_selector, self.feature_selector_filename)
            else:
                X_train_fs = X_train
                X_test_fs = X_test

            for _ in range(num_runs):
                # Shuffle training data to introduce randomness
                indices = np.random.permutation(X_train_fs.shape[0])
                X_train_shuffled = X_train_fs[indices]
                y_train_shuffled = y_train.iloc[indices].reset_index(drop=True)
                # Initialize a new model for each run
                model = MultinomialNB(**self.hyperparams)
                # Train model
                model.fit(X_train_shuffled, y_train_shuffled)
                # Predict on test set
                y_pred = model.predict(X_test_fs)
                # Compute accuracy
                accuracy = accuracy_score(y_test, y_pred)
                self.accuracy_scores.append(accuracy)
            # Save the last trained model
            self.model = model
            # Save the entire class instance
            joblib.dump(self, self.model_filename)


    def predict(self, X_test):
        if self.feature_selector:
            X_test_selected = self.feature_selector.transform(X_test)
            return self.model.predict(X_test_selected)
        else:
            return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self.feature_selector:
            X_test_selected = self.feature_selector.transform(X_test)
            return self.model.predict_proba(X_test_selected)
        else:
            return self.model.predict_proba(X_test)
        

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False, n_jobs=-1, verbose=1):
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"naive_bayes_params_{ngram_type}.pkl")

        if os.path.exists(self.param_filename) and not override:
            loaded_data = joblib.load(self.param_filename)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                best_params, self.num_features = loaded_data
                if isinstance(best_params, str):
                    best_params = ast.literal_eval(best_params)
            else:
                return self.tune_hyperparameters(X, y, ngram_type, override=True)
            
            self.model.set_params(**best_params)
        else:
            param_grid = {
                'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                'num_features': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7078]
            }

            all_params = list(product(param_grid['alpha'], param_grid['num_features']))

            def evaluate_params(alpha, num_features):
                try:
                    # Make a copy of X to ensure it's writable
                    X_copy = X.copy()
                    selector = SelectKBest(chi2, k=min(num_features, X_copy.shape[1]))
                    X_selected = selector.fit_transform(X_copy, y)
                    nb = MultinomialNB(alpha=alpha)
                    scores = cross_val_score(nb, X_selected, y, cv=10, scoring='accuracy', n_jobs=1)
                    score = np.mean(scores)
                except Exception as e:
                    print(f"Error with params alpha={alpha}, num_features={num_features}: {e}")
                    score = -np.inf
                return (score, {'alpha': alpha}, num_features)

            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(evaluate_params)(alpha, num_features) for alpha, num_features in all_params
            )

            best_score = -np.inf
            best_params = {}
            best_num_features = 0
            if results is not None:
                for score, params, num_features in results:
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_num_features = num_features

            if best_params:
                self.model.set_params(**best_params)
                self.num_features = best_num_features
                joblib.dump((best_params, self.num_features), self.param_filename)

    def get_important_features(self):
        """
        Returns the indices of the top features for 'Deceptive' and 'Truthful' classes.
        """
        if self.model is not None:
            if self.feature_selector:
                feature_indices = self.feature_selector.get_support(indices=True)
            else:
                feature_indices = np.arange(self.model.feature_log_prob_.shape[1])
            log_probs = self.model.feature_log_prob_
            diff = log_probs[1] - log_probs[0]
            sorted_indices = np.argsort(diff)[::-1]
            deceptive_indices = feature_indices[sorted_indices[:5]]
            truthful_indices = feature_indices[sorted_indices[-5:]]
            return {
                "Deceptive": deceptive_indices,
                "Truthful": truthful_indices
            }
        else:
            return {"Deceptive": [], "Truthful": []}
