#!/usr/bin/env python3
"""NeuralNetwork class."""
import numpy as np


class NeuralNetwork:
    """Class that defines a neural network with
    one hidden layer performing binary classification"""
    def __init__(self, nx, nodes):
        """Class constructor."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for the private attribute W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter method for the private attribute b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter method for the private attribute A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter method for the private attribute W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter method for the private attribute b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter method for the private attribute A2."""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__A1 = sigmoid(self.__W1 @ X + self.__b1)
        self.__A2 = sigmoid(self.__W2 @ self.__A1 + self.__b2)

        return [self.__A1, self.__A2]

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost /= Y.size

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions."""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)

        predicted = self.__A2.copy()
        predicted[self.__A2 < 0.5] = 0
        predicted[self.__A2 >= 0.5] = 1

        predicted = predicted.astype('int')

        return [predicted, cost]

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network."""
        W2 = self.__W2.copy()

        dZ2 = A2 - Y
        self.__W2 -= alpha * dZ2 @ A1.T / Y.size
        self.__b2 -= alpha * np.sum(dZ2, axis=1, keepdims=True) / Y.size

        # Z1 = self.__W1 @ X + self.__b1
        # dZ1 = (W2.T @ dZ2) * (sigmoid(Z1) * (1 - sigmoid(Z1)))
        dZ1 = (W2.T @ dZ2) * (A1 * (1 - A1))
        self.__W1 -= alpha * (dZ1 @ X.T) / Y.size
        self.__b1 -= alpha * np.sum(dZ1, axis=1, keepdims=True) / Y.size


def sigmoid(z):
    """sigmoid function of np array."""
    sig = 1 / (1 + np.exp(-z))

    return sig
