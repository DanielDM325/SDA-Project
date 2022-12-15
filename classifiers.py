import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


class LogisticRegressionClassifier:
    def __init__(self) -> None:
        self.model = LogisticRegression(solver='sag', max_iter=500)

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        y = self.model.predict(X)
        if len(X.shape) >= 2:
            return y
        else:
            return y[0]


class NaiveBayesClassifier:
    def __init__(self) -> None:
        self.model = GaussianNB()

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        if len(X.shape) >= 2:
            y = self.model.predict(X)
            return y
        else:
            y = self.model.predict([X])
            return y[0]