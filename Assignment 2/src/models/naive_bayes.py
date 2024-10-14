from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
import os
import joblib
import numpy as np

class NaiveBayesClassifier(BaseModel):
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train, ngram_type='uni', override=False):
        """
        Trains the Naive Bayes model on the training data.
        Checks for existing saved weights before training.             
        """
        # Define the weights directory and model filename
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"naive_bayes_model_{ngram_type}.pkl")

        if os.path.exists(self.model_filename) and not override:
            print(f"Loading Naive Bayes model weights from {self.model_filename}")
            self.model = joblib.load(self.model_filename)
        else:
            print("Training Naive Bayes model...")
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.model_filename)
            print(f"Naive Bayes model weights saved to {self.model_filename}")

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False):
        """
        Tunes hyperparameters (alpha for smoothing) using cross-validation.
        Saves the best hyperparameter values.
        """
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"naive_bayes_params_{ngram_type}.pkl")

        if os.path.exists(self.param_filename) and not override:
            print(f"Loading tuned hyperparameters for Naive Bayes from {self.param_filename}")
            best_params = joblib.load(self.param_filename)
            self.model.set_params(**best_params)
        else:
            print("Tuning Naive Bayes model hyperparameters...")
            param_grid = {'alpha': [0.01, 0.1, 1, 10]}
            grid_search = GridSearchCV(self.model, param_grid, cv=5)
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            self.model.set_params(**best_params)
            joblib.dump(best_params, self.param_filename)
            print(f"Tuned hyperparameters saved to {self.param_filename}")

    def get_important_features(self):
        """
        For Naive Bayes, returns the top log probability features for each class.
        Returns the most indicative features for 'fake' and 'genuine' reviews.
        """
        fake_indices = np.argsort(self.model.feature_log_prob_[1])[-5:]
        genuine_indices = np.argsort(self.model.feature_log_prob_[0])[-5:]
        return {
            "Deceptive": fake_indices,
            "Truthful": genuine_indices
        }
