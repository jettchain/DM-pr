from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
import os
import joblib
import numpy as np
class LogisticRegressionClassifier(BaseModel):
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegressionCV(penalty='l1', solver='saga', cv=10, max_iter=5000))
        self.model_filename = None

    def train(self, X_train, y_train, ngram_type='uni', override=False, use_feature_selection=False):
        """
        Trains the Logistic Regression model on the training data.
        Checks for existing saved weights before training.
        """
        # Define the weights directory and model filename
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"logistic_regression_model_{ngram_type}.pkl")

        if os.path.exists(self.model_filename) and not override:
            print(f"Loading Logistic Regression model weights from {self.model_filename}")
            self.model = joblib.load(self.model_filename)
        else:
            print("Training Logistic Regression model...")
            # Note: Logistic Regression doesn't use feature selection in this implementation
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.model_filename)
            print(f"Logistic Regression model weights saved to {self.model_filename}")

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False):
        """
        LogisticRegressionCV handles tuning internally during fitting.
        This method is kept for consistency with the BaseModel interface but does nothing.
        """
        print("LogisticRegressionCV handles tuning internally during fitting. No separate tuning step needed.")
        print("Optimal parameters for Logistic Regression will be determined during training.")

    def get_important_features(self):
        """
        For Logistic Regression, returns the indices of the top absolute coefficient values.
        """
        # Get absolute coefficients for both classes
        coef = self.model.named_steps['logisticregressioncv'].coef_[0]
        fake_indices = np.argsort(coef)[-5:]  # Top 5 positive coefficients (fake)
        genuine_indices = np.argsort(coef)[:5]  # Top 5 negative coefficients (genuine)
        return {
            "Deceptive": fake_indices,
            "Truthful": genuine_indices
        }
