from .base_interface import BaseClass


class SVM(BaseClass):
    """ Implementation of SVM with Stochastic gradient descent """

    def __init__(self, lmbd, D, iterator):
        """
        :param lmbd: (int) value of lambda parameter
        :param D: (int) number of features
        :param iterator: (function) iterator function to go through the dataset by one sample
        """

        self.lmbd = lmbd
        self.D = D + 1
        self.w = [0.] * self.D
        self.iterator = iterator

    def _hinge_loss(self, target, predicted):
        """
        :param target: actual value of target
        :param predicted: predicted value
        :return: value of the hinge loss function
        """

        return max(0, 1 - target * predicted)

    def _sign(self, target):
        """
        :param target: numerical value of target class
        :return: sign of the target
        """

        return -1. if target <= 0 else 1.

    def learn(self, features, targets):
        last = 0

        for t, x, y in self.iterator(features, targets):

            if y == last:
                continue

            alpha = 1. / (self.lmbd * (t + 1.))

            if y * self.infer(x) < 1:

                for i in range(len(x)):
                    self.w[i] = self.w[i] + alpha * ((y * x[i]) + (-2 * self.lmbd * self.w[i]))

            else:
                for i in range(len(x)):
                    self.w[i] = self.w[i] + alpha * (-2 * self.lmbd * self.w[i])
            last = y

    def infer(self, features):
        w_tx = 0.
        for i in range(len(features)):
            w_tx += self.w[i] * features[i]

        return w_tx

    def test(self, features, targets):
        true_negative = 0.
        true_positive = 0.
        total_positive = 0.
        total_negative = 0.
        accuracy = 0.
        loss = 0.
        for _, x, y in self.iterator(features, targets):

            pred = self.infer(x)
            loss += self._hinge_loss(y, pred)
            pred = self._sign(pred)

            if y == 1:
                total_positive += 1.
            else:
                total_negative += 1.

            if pred == y:
                accuracy += 1.
                if pred == 1:
                    true_positive += 1.
                else:
                    true_negative += 1.

        loss = loss / (total_positive + total_negative)
        acc = accuracy / (total_positive + total_negative)
        return loss, acc, true_positive / total_positive, true_negative / total_negative
