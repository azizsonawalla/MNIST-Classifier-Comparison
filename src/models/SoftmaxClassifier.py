import numpy as np
import cupy as cp

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
        print("1")
        XWT = X@W.T  # nxk
        print("2")
        exp_XWT = np.exp(XWT)
        print("3")
        sum_exp_XWT = np.sum(exp_XWT, axis=1)
        print("4")
        logSumExp = np.log(sum_exp_XWT)  # nx1
        print("5")

        XWT_matchingOnly = [XWT[i][y[i]] for i in range(0, n)]  # selects entries from XWT where y_i=c
        XWT_matchingOnly = np.array(XWT_matchingOnly).reshape((n, 1))
        print("6")

        # Calculate the function value
        f = cp.sum(logSumExp - XWT_matchingOnly)

        # Calculate the gradient value
        print("Calculating g")
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

        # Initial guess
        self.w = np.zeros(k*d)
        # utils.check_gradient(self, X, y)
        self.w, _ = utils.findMin(self.funObj, self.w, self.maxEvals, X, y, verbose=self.verbose)
        self.w = np.reshape(self.w, (k, d))

    def predict(self, X):
        return np.argmax(X@self.w.T, axis=1)
