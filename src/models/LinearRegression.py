import numpy as np

import utils


class LinearRegression:

    def __init__(self,
                 loss_function='MSE',
                 polynomial_basis=1,
                 bias=True,
                 regularization_function='L2',
                 lammy=1,
                 max_evals=500):
        """
        Initialize linear regression model
        :param loss_function: the loss function to minimize.
                              'MSE' = Mean Squared Error, 'MAE' = Mean Absolute Error (log sum approximation),
                              'Hub' = Huber loss
        :param polynomial_basis: for change of basis. Performs change of basis for X with powers up to p.
        :param bias: if True, adds col of 1's to the training and test set
        :param regularization_function: function to regularize loss function.
                                        None = no reg., 'L0' = L0 reg., 'L1' = L1 reg., 'L2' = L2 reg.
        :param lammy: coefficient of the regularization function
        :param max_evals: maximum gradient descent evaluations
        """
        self.loss_func = loss_function
        self.p = polynomial_basis
        self.bias = bias
        self.reg_func = regularization_function
        self.lammy = lammy
        self.max_evals = max_evals

    def fit(self, X_train, y_train):

        # perform change of basis (if needed)
        Z = self._perform_change_of_basis(X_train, self.p, bias=self.bias)

        # select regularization_function
        if self.reg_func is None:
            reg_func = self.reg_None
        elif self.reg_func is 'L0':
            reg_func = self.reg_L0
        elif self.reg_func is 'L1':
            reg_func = self.reg_L1
        elif self.reg_func is 'L2':
            reg_func = self.reg_L2
        else:
            return Exception("Invalid regularization function")

        # select the loss function and gradient
        if self.loss_func == 'MSE':
            funObj = self.funObj_MSE
        else:
            return Exception("Invalid loss function")

        # calculate w using gradient descent or SGD
        self.w = np.zeros((Z.shape[1], 1))
        self.w, f = utils.findMin(funObj, self.w, self.max_evals, Z, y_train, reg_func, self.lammy)

    def predict(self, X_test, round=False):
        """
        Predict labels for test samples
        :param X_test: test samples
        :param round: round the labels to the nearest whole number when True
        :return: predicted labels
        """
        # perform change of basis (if needed)
        Z = self._perform_change_of_basis(X_test, self.p, bias=self.bias)

        labels = Z @ self.w
        if round:
            labels = np.round(labels, decimals=0)
        return labels

    @staticmethod
    def _perform_change_of_basis(X, p, bias=True):
        """
        A private helper to perform change of basis and add a bias column
        :param X: original dataset
        :param p: max polynomial exponent of the change of basis
        :param bias: add column of 1's if True
        :return: Z - the transformed dataset
        """

        Z = np.copy(X)

        # if self.p > 1, then perform change of basis
        if p > 1:
            for power in range(2, p+1):
                Z = np.concatenate((Z, X**power), axis=1)

        # if bias=True, then add a column of 1's
        if bias:
            Z = np.insert(Z, 0, np.ones(Z.shape[0]), axis=1)

        return Z

    def funObj_MSE(self, w, X, y, reg_func, lammy):
        """
        Returns the Mean Squared loss for given w, X, y after adding the regularization returned by reg_func
        :param w: params
        :param X: samples
        :param y: actual labels
        :param reg_func: a function that takes ``w`` and returns a regularization term for the loss and gradient
        :return: loss function value, loss function gradient
        """
        reg_f, reg_g = reg_func(w, lammy)  # get the regularization function value and gradient value
        res = (X@w - y)
        f = (res.T@res) + reg_f
        g = X.T@X@w - X.T@y + reg_g
        return f, g

    def funObj_MAE(self, w, X, y, reg_func, lammy):
        """
        Returns the log-sum-exp approximation for the Mean Absolute Error (L1 norm) for given w, X, y after adding
        the regularization returned by reg_func
        :param w: params
        :param X: samples
        :param y: actual labels
        :param reg_func: a function that takes ``w`` and returns a regularization term for the loss and gradient
        :return: loss function value, loss function gradient
        """
        reg_f, reg_g = reg_func(w, lammy)  # get the regularization function value and gradient value

        # Calculate the function value
        r_vals = X @ w - y  # residual values
        f = np.sum([np.log(np.exp(r) + np.exp(-r)) for r in r_vals]) + reg_f  # loss plus regularization

        # Calculate the gradient value
        def get_gradient_wrt(k):  # calculates the partial derivative wrt to w_k
            n = X.shape[0]
            delta_w_k = 0
            for i in range(0, n):
                r = r_vals[i]
                exponent = np.exp(2 * r)
                frac = (exponent - 1) / (exponent + 1)
                x_ik = X[i][k]
                delta_w_k += x_ik * frac
            return delta_w_k
        g = np.array([get_gradient_wrt(k) for k in range(0, X.shape[1])]) + reg_g  # gradient plus regularization grad.

        return f, g

    def funObj_Hub(self, w, X, y, reg_func):
        return NotImplemented("implement this")  # TODO

    def reg_None(self, w, lammy):
        return 0, np.zeros(w.shape)

    def reg_L0(self, w, lammy):
        return NotImplemented("implement this")  # TODO

    def reg_L1(self, w, lammy):
        return NotImplemented("implement this")  # TODO

    def reg_L2(self, w, lammy):
        reg_f = lammy*(w.T@w)
        reg_g = 2*lammy*w
        return reg_f, reg_g




