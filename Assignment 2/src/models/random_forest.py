from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier(oob_score=True, random_state=42)

    def train(self, X_train, y_train):
        """
        Trains the Random Forest model on the training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y):
        """
        Tunes hyperparameters for the Random Forest model, such as the number of trees and max features.
        """
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_features': ['sqrt', 'log2']
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=10)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_

    def get_important_features(self):
        """
        Returns the feature importances from the Random Forest.
        """
        return self.model.feature_importances_
