from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

class LogisticRegressionClassifier(BaseModel):
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegressionCV(penalty='l1', solver='saga', cv=10, max_iter=5000)
        )

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
        return self.model.named_steps['logisticregressioncv'].coef_
