import numpy as np
from scipy.spatial.distance import cdist
from src import utils


class KNN:

    MAX_BLOCK_SIZE = 5000  # maximum number of test samples to process at a time

    def fit(self, X_train, y_train):
        """
        Trains the model on X_train, y_train. This simply stores the data
        :param X_train: training samples
        :param y_train: training sample labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, k_values, metric, verbose=0):
        """
        Predict labels for the test set, using the specified distance function, for each k value in `` k_values ``.

        Note: In this KNN implementation I have added k_values as an array to the predict function (instead of
        initializing the model with a single k value) so that we can compute labels for various k values without
        having to calculate distances again.

        :param X_test: test set to predit labels for
        :param k_values: An array of numbers representing k values. Function will generate an nx1 matrix of predicted
        labels for each k value in this array, and return a set of y_pred matching the same order as `` k_values ``
        :param metric: type of distance function to use. If 'euclidean', then it will use the function in the utils
        package, else parameter is directly passed as `` metric `` to scipy.spatial.distance.cdist
        :return: a list of nx1 y_pred matrices. returned list is of the same size as `` k_values ``. ith matrix in
        returned list is the predicted labels using the ith value in `` k_values `` as number of nearest neighbours.
        """
        T, D = X_test.shape

        y_pred_all = []  # list to store y_pred for each value of k. This will be the list returned
        for _ in k_values:
            y_pred_all.append(np.ones(T))

        # Since KNN calculations involve large arrays, we break it down into 'blocks' to avoid running out of memory
        block_size = min(self.MAX_BLOCK_SIZE, T)
        block_start = 0
        block_end = block_size

        while block_end <= T:

            if verbose > 0:
                print("KNN: calculating labels for test samples {} to {}".format(block_start, block_end-1))

            X_test_block = X_test[block_start:block_end, :]  # extract a 'block' of test samples to predict labels for

            # Calculate an n by block_size matrix of distances between rows of training and test set block
            # using the specified distance function
            if metric == 'euclidean':
                distances = utils.euclidean_dist_squared(self.X_train, X_test_block)
            else:
                distances = cdist(self.X_train, X_test_block, metric)

            for t in range(0, block_end - block_start):

                # get distances from test example t to all training examples i
                distances_to_t = [i[t] for i in distances]

                # sort the list
                indices_of_sorted_distances_to_t = np.argsort(distances_to_t)

                for i in range(0, len(k_values)):
                    k = k_values[i]

                    # pick the K closest training examples
                    k_closest_X_train = indices_of_sorted_distances_to_t[:k]

                    # set y_pred[t] to be the mode of the top K
                    y_closest = self.y_train[k_closest_X_train]
                    y_pred_all[i][t + block_start] = utils.mode(y_closest)

            # update start,end
            if block_end == T:
                break
            block_start = block_end
            block_end = min(T, block_end + block_size)

        return y_pred_all
