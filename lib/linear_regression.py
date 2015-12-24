import numpy as np


class LinearRegression:
    def predict(self, X):
        raise "Implement in subclass"

    def fit(self, X, y):
        raise "Implement in subclass"


class BatchGradient(LinearRegression):
    def __init__(self, settings={'alpha': 0.01, 'iterations': 1500}):
        self.alpha = settings['alpha']
        self.iterations = settings['iterations']

    def predict(self, X):
        x = self._add_ones_to(X)
        return np.dot(x, self.theta)

    def fit(self, X, y):
        n, dim = X.shape

        self.theta = np.zeros(dim + 1)
        x = self._add_ones_to(X)

        for i in range(self.iterations):
            self.theta = self.theta - \
                         (self.alpha / n) * \
                         (x.dot(self.theta) - y).T.dot(x)

    def _add_ones_to(self, X):
        n, dim = X.shape
        x = np.ones([n, dim+1])
        x[: , 1: ] = X

        return x



class LeastSquares(LinearRegression):
    def predict(self, X):
        x = self._add_ones_to(X)

        return x.dot(self.beta)

    def fit(self, X, y):
        x = self._add_ones_to(X)

        self.beta = np.linalg.inv(np.dot(x.T, x)).dot(x.T).dot(y)

    def _add_ones_to(self, X):
        n, dim = X.shape
        x = np.ones([n, dim+1])
        x[: , 1: ] = X

        return x