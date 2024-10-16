from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
import os
import joblib
import numpy as np


class DecisionTreeModel(BaseModel):
    # Set to 10 threads
    def __init__(self):
        self.model = DecisionTreeClassifier()


    def train(self, X_train, y_train, ngram_type='uni', override=False, use_feature_selection=False):
        """
        Trains the Decision Tree model on the training data.
        Checks for existing saved weights before training.
        """
        # Define the weights directory and model filename
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"decision_tree_model_{ngram_type}.pkl")

        if os.path.exists(self.model_filename) and not override:
            print(f"Loading Decision Tree model weights from {self.model_filename}")
            self.model = joblib.load(self.model_filename)
        else:
            print("Training Decision Tree model...")
            # Note: Decision Tree doesn't use feature selection in this implementation
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.model_filename)
            print(f"Decision Tree model weights saved to {self.model_filename}")

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False, n_jobs=-1, verbose=3):
        """
        Tunes hyperparameters for the Decision Tree model using GridSearchCV with cross-validation
        and parallel processing. Saves the best hyperparameter values.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.
            ngram_type (str): Type of n-gram features ("uni" or "bi") for filename differentiation.
            override (bool): If True, performs tuning even if hyperparameter values are already saved.
            n_jobs (int): Number of parallel jobs. -1 means using all available processors.
            verbose (int): Level of verbosity. Higher values increase the amount of messages.
        """
        # Define the weights directory and parameter filename
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"decision_tree_params_{ngram_type}.pkl")

        # Check if the parameter file exists and if override is not set
        if os.path.exists(self.param_filename) and not override:
            try:
                print(f"Loading tuned hyperparameters for Decision Tree from {self.param_filename}")
                best_params = joblib.load(self.param_filename)
                self.model.set_params(**best_params)
                print(f"Optimal parameters for Decision Tree: {best_params}")
                return
            except Exception as e:
                print(f"Failed to load parameters from {self.param_filename}: {e}")
                print("Proceeding to hyperparameter tuning...")

        # Define the hyperparameter grid
        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'ccp_alpha': [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': [None, 'sqrt', 'log2', 0.1, 0.2, 0.5]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=10,  # 10-fold cross-validation
            scoring='accuracy',  # You can choose other metrics like 'f1', 'roc_auc' if needed
            n_jobs=n_jobs,  # Utilize all available cores
            verbose=verbose,  # Increase verbosity for progress monitoring
            return_train_score=False,
            error_score='raise'  # Raise errors encountered during fitting
        )

        print("Starting hyperparameter tuning for Decision Tree...")
        try:
            # Fit GridSearchCV
            grid_search.fit(X, y)
        except Exception as e:
            print(f"An error occurred during GridSearchCV fitting: {e}")
            return

        # Retrieve the best parameters and score
        best_params = grid_search.best_params_
        self.model.set_params(**best_params)

        # Save the best parameters to the parameter file
        try:
            joblib.dump(best_params, self.param_filename)
            print(f"Tuned hyperparameters saved to {self.param_filename}")
            print(f"Optimal parameters for Decision Tree: {best_params}")
        except Exception as e:
            print(f"Failed to save parameters to {self.param_filename}: {e}")

    def get_important_features(self):
        """
        Returns the feature importances from the Decision Tree.
        """
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)
        top_indices = sorted_idx[-5:]  # Top 5 important features
        return {
            "Deceptive": top_indices,
            "Truthful": sorted_idx[:5]  # Bottom 5 important features (as a proxy for genuine)
        }
