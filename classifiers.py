import numpy as np
from sklearn.linear_model import LogisticRegression


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


