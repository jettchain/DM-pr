from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
import os
import joblib
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier()


    def train(self, X_train, y_train, ngram_type='uni', override=False):
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
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.model_filename)
            print(f"Random Forest model weights saved to {self.model_filename}")

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False):
        """
        Tunes hyperparameters for the Random Forest model and saves the best hyperparameter values.
        """
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"random_forest_params_{ngram_type}.pkl")

        if os.path.exists(self.param_filename) and not override:
            print(f"Loading tuned hyperparameters for Random Forest from {self.param_filename}")
            best_params = joblib.load(self.param_filename)
            self.model.set_params(**best_params)
        else:
            print("Tuning Random Forest model hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_features': ['sqrt', 'log2']
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=10)
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            self.model.set_params(**best_params)
            joblib.dump(best_params, self.param_filename)
            print(f"Tuned hyperparameters saved to {self.param_filename}")

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
