from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        """
        Trains the Decision Tree model on the training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y):
        """
        Tunes hyperparameters for the Decision Tree model, such as the cost-complexity pruning parameter alpha.
        """
        param_grid = {'ccp_alpha': [0.0, 0.01, 0.02, 0.05]}
        grid_search = GridSearchCV(self.model, param_grid, cv=10)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_

    def get_important_features(self):
        """
        Returns the feature importances from the Decision Tree.
        """
        return self.model.feature_importances_
