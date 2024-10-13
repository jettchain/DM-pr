from sklearn.naive_bayes import MultinomialNB
from .base_model import BaseModel

class NaiveBayesClassifier(BaseModel):
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        """
        Trains the Naive Bayes model on the training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y):
        """
        Tunes hyperparameters (alpha for smoothing) using cross-validation.
        """
        # Optionally perform a grid search for alpha values
        # Example:
        # from sklearn.model_selection import GridSearchCV
        # param_grid = {'alpha': [0.01, 0.1, 1, 10]}
        # grid_search = GridSearchCV(self.model, param_grid, cv=5)
        # grid_search.fit(X, y)
        # self.model = grid_search.best_estimator_
        pass

    def get_important_features(self):
        """
        Retrieves the feature log probabilities for interpreting important features.
        """
        return self.model.feature_log_prob_
