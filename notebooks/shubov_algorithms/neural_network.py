import numpy as np
import matplotlib.pyplot as plt
from .base_interface import BaseClass


class NeuralNetwork(BaseClass):

    def __init__(self, input_dim, output_dim, hidden_dimension, reg_lambda=0.01, epochs=20000, epsilon=0.001,
                 print_loss=False, print_step=100):
        """
        :param input_dim: number of input dimensions (number of features)
        :param output_dim: number of output dimensions (number of classes)
        :param hidden_dimension: number of neurons in hidden layers
        :param reg_lambda: regularization parameter lambda
        :param epochs: number of epochs
        :param epsilon: epsilon value
        :param print_loss: (boolean) if the current value of the loss function needs to be printed
        :param print_step: number of step when to print the current value of the loss function
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda
        self.hidden_dim = hidden_dimension
        self.epochs = epochs
        self.print_loss = print_loss
        self.epsilon = epsilon
        self.print_step = print_step

        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None

    def calculate_loss(self, x, y):
        """

        :param x: features
        :param y: targets
        :return: loss after forward propagation
        """

        # forward propagation to calculate predictions
        probabilities, _ = self.forward_propagation(x)

        # loss calculation
        correct_log_probabilities = -np.log(probabilities[range(len(x)), y])
        data_loss = np.sum(correct_log_probabilities)

        # loss regularization
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))

        return 1. / len(x) * data_loss

    def forward_propagation(self, x):
        """

        :param x: features
        :return: (1) np array of probabilities for each sample,
            (2) output of the activation function (used for backpropagation)
        """
        z1 = x.dot(self.w1) + self.b1
        a1 = np.tanh(z1)

        z2 = a1.dot(self.w2) + self.b2

        exp_scores = np.exp(z2)

        # SoftMax (output of the final layer)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities, a1

    def infer(self, features):
        probabilities, _ = self.forward_propagation(features)
        return np.argmax(probabilities, axis=1)

    def learn(self, features, targets):
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        self.w1 = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.w2 = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.b2 = np.zeros((1, self.output_dim))

        # This is what we return at the end
        loss_list = []

        # Gradient descent. For each batch...
        for i in range(0, self.epochs):

            # Forward propagation
            probabilities, a1 = self.forward_propagation(features)

            # Back propagation
            delta3 = probabilities
            delta3[range(len(features)), targets] -= 1
            dw2 = a1.T.dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.w2.T) * (1 - np.power(a1, 2))

            dw1 = np.dot(features.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms to weights
            dw2 += self.reg_lambda * self.w2
            dw1 += self.reg_lambda * self.w1

            # Gradient descent parameter update
            self.w1 += -self.epsilon * dw1
            self.b1 += -self.epsilon * db1
            self.w2 += -self.epsilon * dw2
            self.b2 += -self.epsilon * db2

            if self.print_loss and i % self.print_step == 0:
                loss = self.calculate_loss(features, targets)
                print("Loss after iteration %i: %f" % (i, loss))
                loss_list.append(loss)
        if self.print_loss:
            print(plt.plot(loss_list))
