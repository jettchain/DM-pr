from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the provided training data.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Makes predictions on the provided test data.
        """
        pass

    @abstractmethod
    def tune_hyperparameters(self, X, y):
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
