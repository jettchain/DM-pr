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



class NaiveBayesClassifier(BaseModel):

    def __init__(self):
        self.model = MultinomialNB()
        self.feature_selector = None
        self.num_features = None

    def train(self, X_train, y_train, ngram_type='uni', override=False, use_feature_selection=True):
        """
        Trains the Naive Bayes model on the training data.
        Checks for existing saved weights before training.
        Applies feature selection if use_feature_selection is True.
        """
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"naive_bayes_model_{ngram_type}.pkl")
        self.feature_selector_filename = os.path.join(weights_dir, f"naive_bayes_feature_selector_{ngram_type}.pkl")

        if os.path.exists(self.model_filename) and os.path.exists(self.feature_selector_filename) and not override:
            print(f"Loading Naive Bayes model and feature selector from {self.model_filename} and {self.feature_selector_filename}")
            self.model = joblib.load(self.model_filename)
            self.feature_selector = joblib.load(self.feature_selector_filename)
        else:
            print("Training Naive Bayes model...")
            if use_feature_selection and self.num_features is not None:
                self.feature_selector = SelectKBest(chi2, k=self.num_features)
                X_selected = self.feature_selector.fit_transform(X_train, y_train)
                self.model.fit(X_selected, y_train)
            else:
                self.model.fit(X_train, y_train)
            
            joblib.dump(self.model, self.model_filename)
            if self.feature_selector:
                joblib.dump(self.feature_selector, self.feature_selector_filename)
            print(f"Naive Bayes model and feature selector saved to {self.model_filename} and {self.feature_selector_filename}")

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        Applies feature selection if a feature selector is available.
        """
        if self.feature_selector:
            X_test_selected = self.feature_selector.transform(X_test)
            return self.model.predict(X_test_selected)
        else:
            return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False, n_jobs=10, verbose=1):
        """
        Tunes hyperparameters (alpha for smoothing) and number of features using cross-validation.
        Saves the best hyperparameter values and number of features.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.
            ngram_type (str): Type of n-gram features ("uni" or "bi") for filename differentiation.
            override (bool): If True, performs tuning even if hyperparameter values are already saved.
            n_jobs (int): Number of parallel jobs. -1 means using all processors.
            verbose (int): Level of verbosity (0: silent, 1: progress bar).
        """
        # Define the weights directory and parameter filename
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"naive_bayes_params_{ngram_type}.pkl")

        if os.path.exists(self.param_filename) and not override:
            print(f"Loading tuned hyperparameters for Naive Bayes from {self.param_filename}")
            loaded_data = joblib.load(self.param_filename)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                best_params, self.num_features = loaded_data
                if isinstance(best_params, str):
                    best_params = ast.literal_eval(best_params)
            else:
                print("Invalid format for saved parameters. Retuning hyperparameters.")
                return self.tune_hyperparameters(X, y, ngram_type, override=True)
            
            self.model.set_params(**best_params)
            print(f"Optimal parameters for Naive Bayes: alpha={best_params['alpha']}, num_features={self.num_features}")
        else:
            print("Tuning Naive Bayes model hyperparameters with parallel evaluation...")

            # Define the hyperparameter grid
            param_grid = {
                'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                'num_features': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7078]
            }

            # Create all combinations of hyperparameters
            all_params = list(product(
                param_grid['alpha'],
                param_grid['num_features']
            ))

            if verbose:
                print(f"Total hyperparameter combinations to evaluate: {len(all_params)}")

            # Function to evaluate a single set of hyperparameters
            def evaluate_params(alpha, num_features):
                try:
                    # Feature selection
                    selector = SelectKBest(chi2, k=num_features)
                    X_selected = selector.fit_transform(X, y)
                    
                    # Initialize and train the model
                    nb = MultinomialNB(alpha=alpha)
                    scores = cross_val_score(nb, X_selected, y, cv=10, scoring='accuracy', n_jobs=10)
                    score = np.mean(scores)
                except Exception as e:
                    if verbose:
                        print(f"Error with alpha={alpha}, num_features={num_features}: {e}")
                    score = -np.inf  # Assign a poor score in case of failure

                return (score, {'alpha': alpha}, num_features)

            # Use Parallel to evaluate hyperparameters in parallel
            if verbose:
                progress_bar = tqdm(total=len(all_params), desc="Tuning NB Hyperparameters")
            else:
                progress_bar = None

            def update_progress(result):
                if progress_bar:
                    progress_bar.update(1)
                return result

            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(update_progress)(evaluate_params(alpha, num_features)) for alpha, num_features in all_params
            )

            if progress_bar:
                progress_bar.close()

            # Find the best hyperparameters based on cross-validation score
            best_score = -np.inf
            best_params = {}
            best_num_features = 0
            if results is not None:
                for score, params, num_features in results:
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_num_features = num_features
            else:
                print("Parallel execution failed. No results to process.")

            # Set the best parameters and save them
            print(f"Best Cross-Validation Score: {best_score}")
            if best_params:
                print(f"Best Parameters: alpha={best_params['alpha']}, num_features={best_num_features}")
                self.model.set_params(**best_params)
                self.num_features = best_num_features
                joblib.dump((best_params, self.num_features), self.param_filename)
                print(f"Tuned hyperparameters and number of features saved to {self.param_filename}")
                print(f"Optimal parameters for Naive Bayes: alpha={best_params['alpha']}, num_features={self.num_features}")
            else:
                print("No improvement found. Using default parameters.")

    def get_important_features(self):
        """
        For Naive Bayes, returns the top log probability features for each class.
        Returns the most indicative features for 'fake' and 'genuine' reviews.
        """
        if self.feature_selector:
            feature_scores = self.feature_selector.scores_
            top_indices = np.argsort(feature_scores)[-5:]
            return {
                "Deceptive": top_indices,
                "Truthful": np.argsort(feature_scores)[:5]
            }
        else:
            return super().get_important_features()
