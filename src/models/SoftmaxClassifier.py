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

    def funObj(self, W, X, y):

        # TODO: Add option for regularization
        print("Calculating f")

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
        print("Calculating g")
        # g = np.zeros((k, d))
        # for c in range(0, k):
        #     for j in range(0, d):
        #         for i in range(0, n):
        #             I = 1 if y[i] == c else 0
        #             p = exp_XWT[i][c] / sum_exp_XWT[i]
        #             g[c][j] += X[i][j] * (p - I)

        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1

        g = (exp_XWT / sum_exp_XWT[:, None] - y_binary).T @ X
        g = g.flatten()
        return f, g

    def fit(self, X, y):
        n, d = X.shape
        k = np.unique(y).size

        # Initial guess
        self.w = np.zeros(k*d)
        # utils.check_gradient(self, X, y)
        self.w, _ = utils.findMin(self.funObj, self.w, self.maxEvals, X, y, verbose=self.verbose)
        self.w = np.reshape(self.w, (k, d))

    def predict(self, X):
        return np.argmax(X@self.w.T, axis=1)
