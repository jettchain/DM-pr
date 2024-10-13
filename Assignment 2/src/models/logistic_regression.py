from sklearn.linear_model import LogisticRegressionCV
from .base_model import BaseModel

class LogisticRegressionClassifier(BaseModel):
    def __init__(self):
        self.model = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, max_iter=1000)

    def train(self, X_train, y_train):
        """
        Trains the Logistic Regression model on the training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        """
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y):
        """
        Tuning is integrated in LogisticRegressionCV, so no further tuning is necessary.
        """
        # LogisticRegressionCV tunes C internally, no additional tuning needed here
        pass

    def get_important_features(self):
        """
        Retrieves the features with non-zero coefficients as the most important ones.
        """
        return self.model.coef_
