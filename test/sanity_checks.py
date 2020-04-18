# basics
import argparse
import os
import pickle
import sys
# add local modules

sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("..", "src")))
sys.path.append(os.path.abspath(os.path.join("..", "src", "models")))

import numpy as np

# sklearn imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression as skLinearRegression

from src.models.KNN import KNN
from src.models.LinearRegression import LinearRegression as myLinearRegression

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--models', nargs='+', required=True)

    io_args = parser.parse_args()
    models = io_args.models

    def load_dataset(filename):
        with open(os.path.join('test_datasets', filename), 'rb') as f:
            return pickle.load(f)

    errs = ""
    if "KNN" in models:
        print("Running tests for KNN...")
        dataset = load_dataset('citiesSmall.pkl')

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        model_mine = KNN()
        model_mine.fit(X, y)
        k_values = [1, 3, 10]
        metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'cosine', 'euclidean']

        for metric in metrics:
            y_preds_train = model_mine.predict(X, k_values, metric, verbose=1)
            y_preds_test = model_mine.predict(Xtest, k_values, metric, verbose=1)

            for i in range(0, len(k_values)):
                try:
                    model_sklearn = KNeighborsClassifier(k_values[i], metric=metric)
                    model_sklearn.fit(X, y)
                    if not np.array_equal(y_preds_train[i], model_sklearn.predict(X)):
                        errs += "\nKNN: mismatch for training set. K = {}, Metric = {}".format(k_values[i], metric)
                    if not np.array_equal(y_preds_test[i], model_sklearn.predict(Xtest)):
                        errs += "\nKNN: mismatch for test set. K = {}, Metric = {}".format(k_values[i], metric)
                except Exception as e:
                    errs += "\n" + str(e)

    if "LinearRegression" in models:

        print("Running tests for LinearRegression...")

        print(" Testing change of basis")
        test_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_array2 = np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]])
        result = myLinearRegression._perform_change_of_basis(test_array1, 1, True)
        if not np.array_equal(result, test_array2):
            errs += "\nLinear Regression: Failed to perform adding of bias column"
        test_array1 = np.array([[2, 3], [4, 5]])
        test_array2 = np.array([[1, 2, 3, 4, 9, 8, 27], [1, 4, 5, 16, 25, 64, 125]])
        result = myLinearRegression._perform_change_of_basis(test_array1, 3, True)
        if not np.array_equal(result, test_array2):
            errs += "\nLinear Regression: Failed to add polynomial features"


        dataset = load_dataset('basisData.pkl')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        print(" Testing least squares no regularization")
        model_mine = myLinearRegression('MSE', 1, False, None)
        model_mine.fit(X, y)
        model_sklearn = skLinearRegression(False)
        model_sklearn.fit(X, y)
        y_pred_mine = model_mine.predict(X)
        y_pred_sk = model_sklearn.predict(X)
        if not np.array_equal(y_pred_mine, y_pred_sk):
            errs += "\nLinear Regression: did not match sklearn for Least Squares no regularization"
        y_pred_mine = model_mine.predict(X, round=True)
        y_pred_sk = np.round(model_sklearn.predict(X), 0)
        if not np.array_equal(y_pred_mine, y_pred_sk):
            errs += "\nLinear Regression: did not match sklearn for Least Squares no regularization (rounded)"

        # print(" Testing least squares with L2 regularization")
        # model_mine = myLinearRegression('MSE', 1, False, 'L2', 1, 1000)
        # model_mine.fit(X, y)
        # model_sklearn = Ridge(fit_intercept=False)
        # model_sklearn.fit(X, y)
        # y_pred_mine = model_mine.predict(X)
        # y_pred_sk = model_sklearn.predict(X)
        # if not np.array_equal(y_pred_mine, y_pred_sk):
        #     errs += "\nLinear Regression: did not match sklearn for Least Squares L2 regularization"
        #     print(y_pred_mine - y_pred_sk)

    if errs != "":
        print("Some tests failed:")
        print(errs)
    else:
        print(u'\u2713' + " All tests passed ")
