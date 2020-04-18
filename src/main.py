import os
import sys

# add local modules
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("..", "src")))
sys.path.append(os.path.abspath(os.path.join("..", "src", "models")))
from src.models.SoftmaxClassifier import SoftmaxClassifier

import csv
import pickle
import gzip
import argparse

import numpy as np

from sklearn.preprocessing import LabelBinarizer


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "1":

        # ==============================================================================================================
        # Load dataset
        # ==============================================================================================================

        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set

        # use subset
        # subset_size = 20000
        # idx = np.random.choice(X.shape[0], size=subset_size, replace=False)
        # X = X[idx]
        # y = np.reshape(y[idx], (subset_size, 1))

        Xvalid, yvalid = valid_set
        Xtest, ytest = test_set
        n, d = X.shape
        t, _ = Xtest.shape

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)
        Ytest = binarizer.fit_transform(ytest)

        # ==============================================================================================================
        # Helper Functions
        # ==============================================================================================================

        def build_CV_folds(folds, og_X_train, og_y_train):
            """
            Builds the training and testing folds for cross validation from training data
            :param folds: number of folds
            :param og_X_train: original training set samples
            :param og_y_train: original training set labels
            :return: a list of tuples of the form (X_train, y_train, X_test, y_test) where X_test/y_test is n/folds
            randomly selected samples to use as a test set, and X_train/y_train are the remaining samples
            """
            n, d = og_X_train.shape
            all_indices = np.array(range(0, n))                      # indices of all the samples in original dataset
            np.random.shuffle(all_indices)                           # shuffle the indices
            folds_indices_list = np.array_split(all_indices, folds)  # divide the indices into ``folds`` no. of groups

            train_and_test_tuples = []
            for test_fold_indices in folds_indices_list:  # test_fold_indices = indices of samples to use as test set

                # calculate indices of training samples. i.e. indices from 0 to n-1 that are not in test_fold_indices
                train_indices = [i for i in range(0, n) if i not in test_fold_indices]

                # build training and test sets for this fold
                this_X_train = og_X_train[train_indices]
                this_y_train = og_y_train[train_indices]
                this_X_test = og_X_train[test_fold_indices]
                this_y_test = og_y_train[test_fold_indices]

                train_and_test_tuples.append((this_X_train, this_y_train, this_X_test, this_y_test))

            return train_and_test_tuples

        def save_model_results(model_name, csv_row):
            """
            Saves model scores in csv files
            :param model_name: name of the model
            :param csv_row: row of results to add
            """
            results_file_path = os.path.join("CVResults", model_name+".csv")
            with open(results_file_path, mode='a') as results_csv:
                writer = csv.writer(results_csv, delimiter=',')
                writer.writerow(csv_row)

        # ==============================================================================================================
        # Find optimal KNN model
        # ==============================================================================================================

        # print("Running KNN...")
        #
        # # Save headings to results csv
        # save_model_results("KNN", ["K", "Train Error", "Test Error", "Metric"])
        #
        # # Define hyperparameter space
        # k_values = [2**p for p in range(1, 13)]  # k values to test
        # metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'cosine', 'euclidean']  # metrics to test
        #
        # # Initialize and fit model
        # KNN = KNN()
        # KNN.fit(X, y)
        #
        # # for each distance metric
        # for metric in metrics:
        #     print("Metric = " + metric)
        #
        #     # get cross validation folds
        #     cv_datasets = build_CV_folds(3, X, y)
        #
        #     for this_X, this_y, this_Xtest, this_ytest in cv_datasets:
        #
        #         # predicted labels for all k values. y_pred_all_train[i] is predicted labels using k_values[i]
        #         y_preds_all_train = KNN.predict(this_X, k_values, metric, verbose=1)
        #         y_preds_all_test = KNN.predict(this_Xtest, k_values, metric, verbose=1)
        #
        #         for i in range(0, len(k_values)):
        #             y_pred_train = y_preds_all_train[i]
        #             y_pred_test = y_preds_all_test[i]
        #             k = k_values[i]
        #
        #             # calculate scores for this k value, for this fold
        #             this_train_error = np.mean(y_pred_train != this_y)
        #             this_test_error = np.mean(y_pred_test != this_ytest)
        #
        #             save_model_results("KNN", [str(k), str(this_train_error), str(this_test_error), metric])

        # ==============================================================================================================
        # Find optimal Linear Model
        # ==============================================================================================================

        print("Running Linear Model ...")

        softmax = SoftmaxClassifier(verbose=10, maxEvals=100)
        softmax.fit(X, y)

        print("Training error %.3f" % utils.classification_error(softmax.predict(Xtest), ytest))
        # print("Validation error %.3f" % utils.classification_error(softmax.predict(Xtest), ytest))

        # ==============================================================================================================
        # Find optimal SVM
        # ==============================================================================================================


        # TODO: Implement SVM


        # TODO: Implement MLP


        # TODO: Implement CNN

    else:
        print("Unknown question: %s" % question)    