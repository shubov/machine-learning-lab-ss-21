import numpy as np


class BaseClass:
    def learn(self, features: np.array, targets: np.array):
        """
        :param features: numpy array of shape (N, d) with N being the number of samples and d being the number of
            feature dimensions
        :param targets: numpy array of shape (N, 1) with N being the number of samples as in the
            provided features and 1 being the number of target dimensions
        """
        pass

    def infer(self, features):
        """
        :param features: np array of shape (N, d) with N being the number of samples
            and d being the number of feature dimensions
        :return: numpy array of shape (N, 1) of predicted values with N being the number of samples as in the
            provided features and 1 being the number of target dimensions.
        """
        pass
