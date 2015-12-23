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
        return np.dot(self.x, self.theta)

    def fit(self, X, y):
        n, dim = X.shape

        self.theta = np.zeros(dim + 1)
        self.x = np.ones([n, dim+1])
        self.x[: , 1: ] = X

        for i in range(self.iterations):
            self.theta = self.theta - \
                         (self.alpha / n) * \
                         (self.x.dot(self.theta) - y).T.dot(self.x)


class LeastSquares(LinearRegression):
    def predict(self, X):
        return self.x.dot(self.beta)

    def fit(self, X, y):
        n, dim = X.shape
        self.x = np.ones([n, dim+1])
        self.x[: , 1: ] = X

        self.beta = np.linalg.inv(np.dot(self.x.T, self.x)).dot(self.x.T).dot(y)

