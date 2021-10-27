import numpy as np


class LogisticRegression:
    def __init__(
            self,
            learning_rate=0.01,
            epochs=50,
            theta=np.array([1, 1]),
            weights=np.array([-1, 1]),
            intercept=0,
            epsilon: float = 1e-5
    ):
        """
        :param learning_rate: (int) learning rate
        :param epochs: number of epoch (iterations)
        :param theta: (np.array) value of theta parameter
        :param weights: (np.array) initial weights
        :param intercept: (int) value of intercept
        :param epsilon: (float) value of epsilon parameter
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = theta
        self.weights = weights
        self.intercept = intercept
        self.epsilon = epsilon

    def sigmoid(self, weighted_features):
        """
        :param weighted_features: (np.array) weighted features
        :return: sigmoid function
        """
        return 1 / (1 + np.exp(-weighted_features))

    def compute_loss(self, x, y):
        """
        :param x: features
        :param y: predicted targets
        :return: value of the loss function
        """
        batch_size = len(y)
        h = self.sigmoid(np.dot(x, self.theta) + self.intercept)
        loss = (1 / batch_size) * (((-y).T @ np.log(h + self.epsilon)) - ((1 - y).T @ np.log(1 - h + self.epsilon)))
        return loss

    def learn(self, features, targets):
        loss_list = []
        for i in range(self.epochs):
            z = np.dot(features, self.weights)
            y_pred = self.sigmoid(z)
            loss_list.append(self.compute_loss(features, y_pred))
            # D-gradient
            d = np.array([(-2. * np.dot(features[:, j].T, (targets - y_pred))).mean() for j in range(len(features[0]))])
            self.weights = self.weights - self.learning_rate * d
        return loss_list

    def infer(self, features):
        prob = self.sigmoid(np.dot(features, self.weights))
        y_pred = 1 * (prob >= 0.5)
        return y_pred
