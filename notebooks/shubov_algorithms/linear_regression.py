import numpy as np
import numpy.linalg as la
from .base_interface import BaseClass


def get_weights(x, y):
    """ Get initial weights

    :param x: numpy.ndarray of shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
    :param y: array-like of shape (n_samples,)
        Target values.
    :return: numpy.ndarray
        Array of weights
    """

    x_transposed = x.transpose()
    inverse_matrix = np.linalg.inv(x_transposed.dot(x))
    weight = inverse_matrix.dot(x_transposed).dot(y).transpose()

    return weight


class LinearRegression(BaseClass):
    def __init__(self, learning_rate=0.001, epochs=100):
        """

        :param learning_rate: numerical value of learning rate
        :param epochs: number of epochs (iterations)
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scaler = None
        self.weights = None

    def infer(self, features):
        return self._infer(features)

    def _infer(self, features):
        """
        :param features: np array of shape (N, d) with N being the number of samples
            and d being the number of feature dimensions
        :return: numpy array of shape (N, 1) of predicted values with N being the number of samples as in the
            provided features and 1 being the number of target dimensions.
        """
        predict = []
        for i in range(len(features)):
            predict_i = sum(features[i] * self.weights)
            predict.append([predict_i])
        return predict

    def learn(self, features, targets):
        self.weights = get_weights(features, targets)
        y_pred = self._infer(features)

        mse_list = []
        mse = (np.square(targets - y_pred)).mean()

        for i in range(self.epochs):
            mse_list.append(mse)

            y_pred = self._infer(features)

            error = targets - y_pred
            d = np.array([(-2. * np.dot(features[:, j].T, error)).mean() for j in range(len(features[0]))])

            self.weights -= self.learning_rate * d
            y_pred_new = self._infer(features)

            mse = (np.square(targets - y_pred_new)).mean()

        return mse_list
