import gzip
import numpy as np
import os
import pickle
from numpy.linalg import norm
from scipy.optimize import approx_fprime

from scipy import stats

def load_dataset(dataset_name):

    # Load and standardize the data and add the bias term
    if dataset_name == "logisticData":
        with open(os.path.join('..', 'data', 'logisticData.pkl'), 'rb') as f:
            data = pickle.load(f)

        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        return {"X":X, "y":y,
                "Xvalid":Xvalid,
                "yvalid":yvalid}

    elif dataset_name == "multiData":
        with open(os.path.join('data', 'multiData.pkl'), 'rb') as f:
            data = pickle.load(f)

        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        y -= 1  # so classes are 0,..,4
        yvalid -=1

        return {"X":X, "y":y,
                "Xvalid":Xvalid,
                "yvalid":yvalid}

    else:
        with open(os.path.join('..', DATA_DIR, dataset_name+'.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data


def load_dataset_multi(dataset_name):
        with gzip.open(os.path.join('data', dataset_name), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        # X, mu, sigma = standardize_cols(X)
        # Xtest, _, _ = standardize_cols(Xtest, mu, sigma)
        #
        # X = np.hstack([np.ones((X.shape[0], 1)), X])
        # Xtest = np.hstack([np.ones((Xtest.shape[0], 1)), Xtest])

        # y -= 1  # so classes are 0,..,4
        # yvalid -=1

        return {"X": X,
                "y": y,
                "Xtest": Xtest,
                "ytest": ytest}


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')


def classification_error(y, yhat):
    return np.mean(y!=yhat)


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following src will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)

    # without broadcasting:
    # n,d = X.shape
    # t,d = Xtest.shape
    # D = X**2@np.ones((d,t)) + np.ones((n,d))@(Xtest.T)**2 - 2*X@Xtest.T


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y) == 0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


def findMin(funObj, w, maxEvals, *args, verbose=0):
    """
    Uses gradient descent to optimize the objective function

    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    funEvals = 1

    alpha = 0.1
    while True:
        # Line-search using quadratic interpolation to find an acceptable value of alpha
        gg = g.T.dot(g)

        while True:
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, *args)

            funEvals += 1

            if f_new <= f - gamma * alpha*gg:
                break

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # Update step size alpha
            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

        # Print progress
        if verbose > 0:
            print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        if verbose > 1:
            print("alpha: %.3f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w, f