import os
import joblib
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, ngram_type='uni', override=False, use_feature_selection=False):
        """
        Trains the model on the provided training data.
        If saved weights exist, loads them instead of retraining.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Makes predictions on the provided test data.
        """
        pass

    @abstractmethod
    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False):
        """
        Tunes hyperparameters for the model using cross-validation.
        """
        pass

    @abstractmethod
    def get_important_features(self):
        """
        Returns the most important features for interpretation.
        """
        pass
