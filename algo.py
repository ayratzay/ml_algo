__author__ = 'Freeman'

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


def linear_regression(x0, x1, theta0, theta1):
    return x0 * theta0 + x1 * theta1


def squared_delta(h_x, y):
    return (h_x - y) ** 2


def mse(ds, t0, t1):
    sqe = [squared_delta(linear_regression(x0, x1, t0, t1), y) for (x0, x1, y) in ds]
    return sum(sqe) / (2 * len(sqe))


def derivative_gradient(h_x, y, der_x):
    return (h_x - y) * der_x


def derivative_delta(ds, t0, t1, x_index):
    gd = [derivative_gradient(linear_regression(x0, x1, t0, t1), y, (x0, x1)[x_index]) for (x0, x1, y) in ds]
    return sum(gd) / len(gd)


class LinerRegressionNoMatrix:
    def __init__(self, learning_rate=0.05, iterations=1000):
        self.learning_rate = learning_rate
        self.theta = None
        self.iterations = iterations
        self.mse = None

    def __repr__(self):
        return ("linear regression model")

    def fit(self, ds):
        i = 0

        # print (t0, t1, mse(dataset, t0, t1))
        t0, t1 = random.randint(0, 10), random.randint(0, 10)
        while i < self.iterations:
            t0, t1 = t0 - self.learning_rate * derivative_delta(ds, t0, t1,
                                                                0), t1 - self.learning_rate * derivative_delta(ds, t0,
                                                                                                               t1, 1)
            i += 1
        self.theta = (t0, t1)
        self.mse = mse(ds, t0, t1)

    def predict(self, x):
        return linear_regression(1, x, *self.theta)

    def print_mse(self):
        print ("MSE is " + str(self.mse))


class LinearRegressionMatrix:
    def __init__(self, learning_rate=0.05, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self._theta = None
        self._mse = None

    def __repr__(self):
        return "Linear regression based on matricies"

    def _scalar_normalization(self, X):
        mean_v = X.mean(axis=0)
        len_v = X.max(axis = 0) - X.min(axis = 0)
        mean_v[0], len_v[0] = 0, 1
        X_new = (X - mean_v)/ len_v
        return X_new

    def fit(self, X, y):
        global hist
        X = self._scalar_normalization(X)
        theta = np.random.randint(10, size=X.shape[1])
        sample_size = X.shape[0]

        mse_prev = None
        for i in xrange(self.iterations):
            error_array = np.dot(X, theta) - y
            mse = sum(error_array ** 2 / (2 * sample_size))
            if mse_prev:
                if (mse_prev - mse) < 0.001:
                    break
            mse_prev = mse
            theta = theta - theta * (self.learning_rate * np.dot(error_array.T, X) / sample_size)
        hist[self.learning_rate] = (i, mse)
        self._mse = mse
        self._theta = theta

    def predict(self, x):
        return  sum(x * self._theta)


class NormalEquation:

    def __init__(self, ):
        self._mse = None
        self._theta = None

    def __repr__(self):
        return "Solving equation with normal equation"

    def solve(self,X, y):
        sq = np.dot(X.T, X)
        inv = np.linalg.inv(sq)
        step = np.dot(inv, X.T)
        self._theta = np.dot(step, y)
        error_array = np.dot(X, self._theta) - y
        self._mse = sum(error_array ** 2 / (2 * X.shape[0]))


class LogisticRegressionMatrix:
    def __init__(self, learning_rate=0.05, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self._theta = None
        self._mse = None

    def __repr__(self):
        return "Logistic regression based on matricies"

    def fit(self, X, y):
        m = X.shape[0]
        theta = np.random.randint(m, size=X.shape[1])

        for i in xrange(self.iterations):
            hypothesis = expit(X.dot(theta))
            term1 = y * np.log(hypothesis)
            term2 = (1 - y) * np.log(1 - hypothesis)
            cost = sum((term1 + term2) / -m)
            theta = theta - self.learning_rate * X.T.dot( (hypothesis - y))
            print cost
        self._theta = theta

    def predict(self, x, boolean = True):
        if boolean:
            return (expit(x.dot(self._theta)) >= 0.5).astype(int)
        else:
            return expit(x.dot(self._theta))

