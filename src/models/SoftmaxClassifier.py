import numpy as np
import os
import sys
# add local modules
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("..", "..")))
sys.path.append(os.path.abspath(os.path.join("..", "..", "..")))

from src import utils


class SoftmaxClassifier:

    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes
        W = np.reshape(w, (k, d))
        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1
        XW = np.dot(X, W.T)
        Z = np.sum(np.exp(XW), axis=1)
        # Calculate the function value
        f = - np.sum(XW[y_binary] - np.log(Z))
        # Calculate the gradient value
        g = (np.exp(XW) / Z[:, None] - y_binary).T @ X
        return f, g.flatten()

    def funObj2(self, W, X, y):

        # TODO: Add option for regularization

        n, d = X.shape
        k = np.unique(y).size

        W = W.reshape(k, d)
        XWT = X@W.T  # nxk
        exp_XWT = np.exp(XWT)
        sum_exp_XWT = np.sum(exp_XWT, axis=1)
        logSumExp = np.log(sum_exp_XWT)  # nx1

        XWT_matchingOnly = [XWT[i][y[i]] for i in range(0, n)]  # selects entries from XWT where y_i=c
        XWT_matchingOnly = np.array(XWT_matchingOnly).reshape((n, 1))

        # Calculate the function value
        f = np.sum(logSumExp - XWT_matchingOnly)

        # Calculate the gradient value
        g = np.zeros((k, d))
        for c in range(0, k):
            for j in range(0, d):
                for i in range(0, n):
                    I = 1 if y[i] == c else 0
                    p = exp_XWT[i][c] / sum_exp_XWT[i]
                    g[c][j] += X[i][j] * (p - I)
        g = g.flatten()
        return f, g

    def fit(self, X, y):
        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k
        self.W = np.zeros(d * k)
        self.w = self.W  # because the gradient checker is implemented in a silly way
        # Initial guess
        # utils.check_gradient(self, X, y)
        (self.W, f) = utils.findMin(self.funObj, self.W,
                                      self.maxEvals, X, y, verbose=self.verbose)
        self.W = np.reshape(self.W, (k, d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
