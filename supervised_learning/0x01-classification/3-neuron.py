#!/usr/bin/env python3
"""Class Neuron."""
import numpy as np


class Neuron:
    """Class Neuron that defines a single neuron
    performing binary classification."""
    def __init__(self, nx):
        """Class constructor."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for W private attribute."""
        return self.__W
    @property
    def b(self):
        """getter for b private attribute."""
        return self.__b
    @property
    def A(self):
        """getter for A private attribute."""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of a neuron."""
        W = self.__W
        b = self.__b

        self.__A = sigmoid((W @ X) + b)

        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost /= Y.size

        return cost


def sigmoid(z):
    """sigmoid function of np array."""
    sig = 1 / (1 + np.exp(-z))

    return sig
