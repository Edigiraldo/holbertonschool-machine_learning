#!/usr/bin/env python3
"""Class Neuron."""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions and cost."""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predicted = A.copy()
        predicted[A < 0.5] = 0
        predicted[A >= 0.5] = 1

        return [predicted, cost]

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron."""
        self.__W -= alpha * np.sum((A - Y) * X, axis=1) / Y.size
        self.__b -= alpha * np.sum((A - Y)) / Y.size

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Method that trains the neuron."""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        iterat = []
        costs = []
        A = self.forward_prop(X)
        iterat.append(0)
        costs.append(self.cost(Y, A))
        if verbose:
            print("Cost after", 0, "iterations:", costs[-1])

        for i in range(1, iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                iterat.append(i)
                costs.append(self.cost(Y, A))
                if verbose:
                    print("Cost after", i, "iterations:", costs[-1])

        if graph:
            plt.plot(iterat, costs, '-b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)


def sigmoid(z):
    """sigmoid function of np array."""
    sig = 1 / (1 + np.exp(-z))

    return sig
