import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from data_analysis import find_distribution
from scipy import stats


class LogisticRegressionClassifier:
    """
    A logistic regression classifier based on the module in sklearn:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Simply feed the data X and y into the fit method and predict using the predict method.
    """
    def __init__(self) -> None:
        self.model = LogisticRegression(solver='sag', max_iter=100000, class_weight='balanced')

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        y = self.model.predict(X)
        if len(X.shape) >= 2:
            return y
        else:
            return y[0]


class NaiveBayesClassifier:
    """
    A Naive Bayes classifier based on the module in sklearn:
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
    Simply feed the data X and y into the fit method and predict using the predict method.
    It uses a Gaussian distribution to estimate features irregardless if these
    features are Gaussian or not.
    """
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


class MultiNaiveBayesClassifier:
    """
    Like the Naive Bayes classifier from sklearn it uses Bayesian statistics
    to classify. The Gaussian sklearn version assumes all distributions are
    distributed according to a Gaussian distribution. This custom
    implementation does not. It uses the find_distritution function from
    data_analysis.py (see comments for that function to further understand
    function) to find the best distribution for every feature. For
    distributions which are boolean a Bernoulli distribution is used. And for
    distributions which are categorical the same probility function is used as
    the one in sklearn:
    https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes
    With the same alpha value of 1.0.
    As a class it works the same as the sklearn version. Simply feed the data
    X and y into fit and use predict to classify. Except additionaly a list
    of distributions to consider for finding the best distribution for every
    relevant feature can be provided when creating the class.
    It must be noted as with the find_distribution function training this
    model requires significant more computation than a simple Naive Bayes from
    sklearn.
    """
    def __init__(self, alpha: float = 1.0, distributions_consider: list = list()) -> None:
        self.alpha = alpha
        self.P_1 = 0.0
        self.P_0 = 0.0
        self.distributions_1 = list()
        self.distributions_0 = list()
        self.distributions_consider = distributions_consider

    def fit(self, X, y) -> None:
        X = np.array(X)
        y = np.array(y)
        self.P_1 = np.sum(y) / y.size
        self.P_0 = 1.0 - self.P_1
        X_1 = X[np.argwhere(y).flatten()]
        X_0 = X[np.argwhere(y == 0).flatten()]
        for column in X_1.T:
            print(column)
            if np.unique(column).size == 2:
                bernoulli = dict()
                for value in np.unique(column):
                    bernoulli[value] = (column == value).sum() / column.size
                self.distributions_1.append(bernoulli)
            elif column.dtype == object:
                categorical = dict()
                for value in np.unique(column):
                    categorical[value] = ((column == value).sum() + self.alpha) / (column.size + self.alpha * np.unique(column).size)
                self.distributions_1.append(categorical)
            else:
                print(best_distribution[0])
                best_distribution = find_distribution(column, self.distributions_consider)
                self.distributions_1.append((best_distribution[0], best_distribution[1]))
        for column in X_0.T:
            print(column)
            if np.unique(column).size == 2:
                bernoulli = dict()
                for value in np.unique(column):
                    bernoulli[value] = (column == value).sum() / column.size
                self.distributions_0.append(bernoulli)
            elif column.dtype == object:
                categorical = dict()
                for value in np.unique(column):
                    categorical[value] = ((column == value).sum() + self.alpha) / (column.size + self.alpha * np.unique(column).size)
                self.distributions_0.append(categorical)
            else:
                print(best_distribution[0])
                best_distribution = find_distribution(column, self.distributions_consider)
                self.distributions_0.append((best_distribution[0], best_distribution[1]))

    def predict(self, X) -> np.array:
        predictions = list()
        for sample in X:
            posterior_1 = self.P_1
            posterior_0 = self.P_0
            for distribution, feature in enumerate(sample):
                if type(self.distributions_1[distribution]) == dict:
                    posterior_1 *= self.distributions_1[distribution][X]
                    posterior_0 *= self.distributions_0[distribution][X]
                else:
                    paramaters = self.distributions_1[distribution][1]
                    args = paramaters[:-2]
                    mean = paramaters[-2]
                    standard_deviation = paramaters[-1]
                    if args:
                        posterior_1 *= getattr(stats, self.distributions_1[distribution][0]).pdf(feature, *args, loc=mean, scale=standard_deviation)
                    else:
                        posterior_1 *= getattr(stats, self.distributions_1[distribution][0]).pdf(feature, loc=mean, scale=standard_deviation)

                    paramaters = self.distributions_0[distribution][1]
                    args = paramaters[:-2]
                    mean = paramaters[-2]
                    standard_deviation = paramaters[-1]
                    if args:
                        posterior_0 *= getattr(stats, self.distributions_0[distribution][0]).pdf(feature, *args, loc=mean, scale=standard_deviation)
                    else:
                        posterior_0 *= getattr(stats, self.distributions_0[distribution][0]).pdf(feature, loc=mean, scale=standard_deviation)
            if posterior_1 > posterior_0:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)


class KNearestNeighborsClassifier:
    def __init__(self) -> None:
        self.model = KNeighborsClassifier()

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        if len(X.shape) >= 2:
            y = self.model.predict(X)
            return y
        else:
            y = self.model.predict([X])
            return y[0]
