from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegressionClassifier(BaseModel):
    def __init__(self):
        self.model = LogisticRegressionCV()
        self.scaler = None  # To store the scaler if used
        self.accuracy_scores = []
        self.hyperparams = None
        self.model_filename = None

    def train(self, X_train, y_train, X_test, y_test, ngram_type='uni', num_runs=50, override=False):
        # Define directories and filenames
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"logistic_regression_model_{ngram_type}.pkl")
        hyperparams_filename = os.path.join(weights_dir, f"logistic_regression_params_{ngram_type}.pkl")
        scaler_filename = os.path.join(weights_dir, f"logistic_regression_scaler_{ngram_type}.pkl")

                  
        # Load the entire class instance if available
        if os.path.exists(self.model_filename) and not override:
            # Load the entire class instance
            loaded_self = joblib.load(self.model_filename)
            self.__dict__.update(loaded_self.__dict__)
        else:
            self.accuracy_scores = []
            # Initialize the scaler
            self.scaler = StandardScaler(with_mean=False)
            # Scale the training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            # Save the scaler
            joblib.dump(self.scaler, scaler_filename)
            for _ in range(num_runs):
                # Initialize the Logistic Regression model
                model = LogisticRegressionCV(random_state=np.random.randint(0, 10000))
                # Train the model
                model.fit(X_train_scaled, y_train)
                # Predict on test set
                y_pred = model.predict(X_test_scaled)
                # Compute accuracy
                accuracy = accuracy_score(y_test, y_pred)
                self.accuracy_scores.append(accuracy)
                self.model = model  # Save the last trained model

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

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False):
        """
        LogisticRegressionCV handles tuning internally during fitting.
        This method is kept for consistency with the BaseModel interface but does nothing.
        """
        print("LogisticRegressionCV handles tuning internally during fitting. No separate tuning step needed.")
        print("Optimal parameters for Logistic Regression will be determined during training.")


    def get_important_features(self):
        """Returns indices of the most important features for both classes."""
        if self.model is not None:
            # Access coefficients directly from LogisticRegressionCV
            coef = self.model.coef_[0]  # Get coefficients for the positive class
            indices = np.argsort(coef)

            deceptive_indices = indices[-5:]  # Top 5 positive coefficients (indicative of deceptive reviews)
            truthful_indices = indices[:5]    # Top 5 negative coefficients (indicative of truthful reviews)

            return {
                "Deceptive": deceptive_indices.tolist(),
                "Truthful": truthful_indices.tolist()
            }
        else:
            return {"Deceptive": [], "Truthful": []}


