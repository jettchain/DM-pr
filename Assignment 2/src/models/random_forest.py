from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
import os
import joblib
from joblib import Parallel, delayed
import numpy as np
from itertools import product

class RandomForestModel(BaseModel):
    os.environ['NUMEXPR_MAX_THREADS'] = '10'  # Set to 10 threads
    def __init__(self):
        self.model = RandomForestClassifier(oob_score=True)


    def train(self, X_train, y_train, ngram_type='uni', override=False, use_feature_selection=False):
        """
        Trains the Random Forest model on the training data.
        Checks for existing saved weights before training.
        """
        # Define the weights directory and model filename
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"random_forest_model_{ngram_type}.pkl")

        if os.path.exists(self.model_filename) and not override:
            print(f"Loading Random Forest model weights from {self.model_filename}")
            self.model = joblib.load(self.model_filename)
        else:
            print("Training Random Forest model...")
            # Note: Random Forest doesn't use feature selection in this implementation
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.model_filename)
            print(f"Random Forest model weights saved to {self.model_filename}")

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False, n_jobs=-1, verbose=1):
        """
        Tunes hyperparameters for the Random Forest model using Out of Bag (OOB) evaluation
        and saves the best hyperparameter values.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.
            ngram_type (str): Type of n-gram features ("uni" or "bi") for filename differentiation.
            override (bool): If True, performs tuning even if hyperparameter values are already saved.
            n_jobs (int): Number of parallel jobs. -1 means using all processors.
            verbose (int): Level of verbosity.
        """
        # Define the weights directory and parameter filename
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"random_forest_params_{ngram_type}.pkl")

        if os.path.exists(self.param_filename) and not override:
            print(f"Loading tuned hyperparameters for Random Forest from {self.param_filename}")
            best_params = joblib.load(self.param_filename)
            self.model.set_params(**best_params)
            print(f"Optimal parameters for Random Forest: {best_params}")
        else:
            print("Tuning Random Forest model hyperparameters with parallel OOB evaluation...")

            param_grid = {
                'n_estimators': [50, 100, 200, 300, 400, 500, 750, 1000, 1500],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'max_features': ['sqrt', 'log2', None, 0.1, 0.2, 0.5],
                'bootstrap': [True]  # Required for OOB
            }

            # Create all combinations of hyperparameters
            all_params = list(product(*param_grid.values()))

            if verbose:
                print(f"Total hyperparameter combinations to evaluate: {len(all_params)}")

            # Function to evaluate a single set of hyperparameters
            def evaluate_params(params):
                rf = RandomForestClassifier(oob_score=True, n_jobs=1, **dict(zip(param_grid.keys(), params)))
                try:
                    rf.fit(X, y)
                    oob_score = rf.oob_score_
                except Exception as e:
                    if verbose:
                        print(f"Error with params {params}: {e}")
                    oob_score = -1
                return (oob_score, dict(zip(param_grid.keys(), params)))

            # Use Parallel to evaluate hyperparameters in parallel
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(evaluate_params)(params) for params in all_params
            )

            if results:
                best_params = max(results, key=lambda x: x[0])[1]
                best_oob_score = max(results, key=lambda x: x[0])[0]
            else:
                print("Hyperparameter tuning failed. Using default parameters.")
                best_params = None
                best_oob_score = -1

            # Set the best parameters and save them
            print(f"Best OOB Score: {best_oob_score}")
            if best_params:
                print("Best Parameters:", best_params)
                self.model.set_params(**best_params)
                joblib.dump(best_params, self.param_filename)  
                print(f"Tuned hyperparameters saved to {self.param_filename}")
                print(f"Optimal parameters for Random Forest: {best_params}")
            else:
                print("No improvement found. Using default parameters.")

    def get_important_features(self):
        """
        Returns the feature importances from the Random Forest.
        """
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)
        top_indices = sorted_idx[-5:]  # Top 5 important features
        return {
            "Deceptive": top_indices,
            "Truthful": sorted_idx[:5]  # Bottom 5 important features (as a proxy for genuine)
        }
