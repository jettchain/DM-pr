from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
import os
import joblib
from joblib import Parallel, delayed
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier()
        self.accuracy_scores = []
        self.hyperparams = None
        self.model_filename = None

    def train(self, X_train, y_train, X_test, y_test, ngram_type='uni', num_runs=50, override=False):
        # Define directories and filenames
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"random_forest_model_{ngram_type}.pkl")
        hyperparams_filename = os.path.join(weights_dir, f"random_forest_params_{ngram_type}.pkl")

        # Load hyperparameters if available
        if self.hyperparams is None:
            if os.path.exists(hyperparams_filename):
                self.hyperparams = joblib.load(hyperparams_filename)
            else:
                raise ValueError("Hyperparameters not found. Please run tune_hyperparameters before training.")

        # Load the entire class instance if available
        if os.path.exists(self.model_filename) and not override:
            # Load the entire class instance
            loaded_self = joblib.load(self.model_filename)
            self.__dict__.update(loaded_self.__dict__)
        else:
            self.accuracy_scores = []
            for _ in range(num_runs):
                # Initialize a new model for each run
                model = RandomForestClassifier(**self.hyperparams, random_state=np.random.randint(0, 10000))
                # Train the model
                model.fit(X_train, y_train)
                # Predict on test set
                y_pred = model.predict(X_test)
                # Compute accuracy
                accuracy = accuracy_score(y_test, y_pred)
                self.accuracy_scores.append(accuracy)
                self.model = model  # Save the last model

            # Save the entire class instance
            joblib.dump(self, self.model_filename)


    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Predicts class probabilities for the test data.
        """
        return self.model.predict_proba(X_test)

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
        if self.model is not None:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            deceptive_indices = indices[:5]
            truthful_indices = indices[-5:]
            return {
                "Deceptive": deceptive_indices,
                "Truthful": truthful_indices
            }
        else:
            return {"Deceptive": [], "Truthful": []}
