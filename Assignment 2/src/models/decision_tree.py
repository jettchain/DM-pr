from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
import os
import joblib
import numpy as np

class DecisionTreeModel(BaseModel):
    def __init__(self):
        self.model = DecisionTreeClassifier()


    def train(self, X_train, y_train, ngram_type='uni', override=False):
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
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.model_filename)
            print(f"Decision Tree model weights saved to {self.model_filename}")

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False):
        """
        Tunes hyperparameters for the Decision Tree model and saves the best hyperparameter values.
        """
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"decision_tree_params_{ngram_type}.pkl")

        if os.path.exists(self.param_filename) and not override:
            print(f"Loading tuned hyperparameters for Decision Tree from {self.param_filename}")
            best_params = joblib.load(self.param_filename)
            self.model.set_params(**best_params)
        else:
            print("Tuning Decision Tree model hyperparameters...")
            param_grid = {'ccp_alpha': [0.0, 0.01, 0.02, 0.05]}
            grid_search = GridSearchCV(self.model, param_grid, cv=10)
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            self.model.set_params(**best_params)
            joblib.dump(best_params, self.param_filename)
            print(f"Tuned hyperparameters saved to {self.param_filename}")

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
