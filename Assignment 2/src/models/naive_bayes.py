from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from .base_model import BaseModel
import os
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
import ast 
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product



class NaiveBayesClassifier(BaseModel):

    def __init__(self):
        self.model = MultinomialNB()
        self.feature_selector = None
        self.num_features = None

    def train(self, X_train, y_train, ngram_type='uni', override=False, use_feature_selection=True):
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.model_filename = os.path.join(weights_dir, f"naive_bayes_model_{ngram_type}.pkl")
        self.feature_selector_filename = os.path.join(weights_dir, f"naive_bayes_feature_selector_{ngram_type}.pkl")

        if os.path.exists(self.model_filename) and os.path.exists(self.feature_selector_filename) and not override:
            self.model = joblib.load(self.model_filename)
            self.feature_selector = joblib.load(self.feature_selector_filename)
        else:
            if use_feature_selection and self.num_features is not None:
                self.feature_selector = SelectKBest(chi2, k=self.num_features)
                X_selected = self.feature_selector.fit_transform(X_train, y_train)
                self.model.fit(X_selected, y_train)
            else:
                self.model.fit(X_train, y_train)
            
            joblib.dump(self.model, self.model_filename)
            if self.feature_selector:
                joblib.dump(self.feature_selector, self.feature_selector_filename)

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        Applies feature selection if a feature selector is available.
        """
        if self.feature_selector:
            X_test_selected = self.feature_selector.transform(X_test)
            return self.model.predict(X_test_selected)
        else:
            return self.model.predict(X_test)

    def tune_hyperparameters(self, X, y, ngram_type='uni', override=False, n_jobs=-1, verbose=1):
        weights_dir = "Assignment 2/src/models/weights"
        os.makedirs(weights_dir, exist_ok=True)
        self.param_filename = os.path.join(weights_dir, f"naive_bayes_params_{ngram_type}.pkl")

        if os.path.exists(self.param_filename) and not override:
            loaded_data = joblib.load(self.param_filename)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                best_params, self.num_features = loaded_data
                if isinstance(best_params, str):
                    best_params = ast.literal_eval(best_params)
            else:
                return self.tune_hyperparameters(X, y, ngram_type, override=True)
            
            self.model.set_params(**best_params)
        else:
            param_grid = {
                'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                'num_features': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7078]
            }

            all_params = list(product(param_grid['alpha'], param_grid['num_features']))

            def evaluate_params(alpha, num_features):
                try:
                    selector = SelectKBest(chi2, k=num_features)
                    X_selected = selector.fit_transform(X, y)
                    nb = MultinomialNB(alpha=alpha)
                    scores = cross_val_score(nb, X_selected, y, cv=10, scoring='accuracy', n_jobs=n_jobs)
                    score = np.mean(scores)
                except Exception as e:
                    score = -np.inf
                return (score, {'alpha': alpha}, num_features)

            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(evaluate_params)(alpha, num_features) for alpha, num_features in all_params
            )

            best_score = -np.inf
            best_params = {}
            best_num_features = 0
            if results is not None:
                for score, params, num_features in results:
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_num_features = num_features

            if best_params:
                self.model.set_params(**best_params)
                self.num_features = best_num_features
                joblib.dump((best_params, self.num_features), self.param_filename)

    def get_important_features(self):
        """
        For Naive Bayes, returns the top log probability features for each class.
        Returns the most indicative features for 'fake' and 'genuine' reviews.
        """
        if self.feature_selector:
            feature_scores = self.feature_selector.scores_
            top_indices = np.argsort(feature_scores)[-5:]
            return {
                "Deceptive": top_indices,
                "Truthful": np.argsort(feature_scores)[:5]
            }
        else:
            return super().get_important_features()
