#!/usr/bin/env python3
"""DeepNeuralNetwork class."""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """class that defines a deep neural network
       performing binary classification."""
    def __init__(self, nx, layers):
        """Class constructor."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")

        # A dictionary to hold all intermediary values of the network.
        self.__cache = {}
        # A dictionary to hold all weights and biased of the network.
        self.__weights = {}

        for i in range(len(layers)):

            val = layers[i]
            if type(val) != int or val < 1:
                raise ValueError("layers must be a list of positive integers")

            nameW = 'W' + str(i + 1)
            nameb = 'b' + str(i + 1)

            neurons = layers[i]
            if i != 0:
                weights = layers[i - 1]
            else:
                weights = nx

            self.__weights[nameW] = np.random.randn(neurons,
                                                    weights)*np.sqrt(2/weights)
            self.__weights[nameb] = np.zeros((neurons, 1))

        # The number of layers in the neural network.
        self.__L = len(layers)

    @property
    def L(self):
        """Getter method for attribute L."""

        return self.__L

    @property
    def cache(self):
        """Getter method for attribute cache."""

        return self.__cache

    @property
    def weights(self):
        """Getter method for attrubute weights."""

        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        n_layers = self.__L
        cache = self.__cache
        weights = self.__weights

        cache['A0'] = X

        for i in range(n_layers):
            inputs = cache['A' + str(i)]
            weight = weights['W' + str(i + 1)]
            bias = weights['b' + str(i + 1)]

            cache['A' + str(i + 1)] = sigmoid(weight @ inputs + bias)

        return [cache['A' + str(i + 1)], cache]

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost /= Y.size

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        predict = A.copy()
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 1

        predict = predict.astype('int')

        return [predict, cost]

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        n_layers = self.__L

        for i in range(n_layers, 0, -1):
            Il = cache['A' + str(i - 1)]  # Input for layer
            Wl = self.__weights['W' + str(i)]
            bl = self.__weights['b' + str(i)]
            Al = cache['A' + str(i)]

            if i != n_layers:
                dZl = (Wnl.T @ dZnl) * (Al * (1 - Al))
            else:
                Aout = cache['A' + str(n_layers)]
                dZl = Aout - Y

            # In next iteration current W's are W's from next layer.
            Wnl = Wl.copy()

            Wl -= alpha * (dZl @ Il.T) / Y.size
            bl -= alpha * np.sum(dZl, axis=1, keepdims=True) / Y.size

            dZnl = dZl

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network.""" 
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
        Aout, cache = self.forward_prop(X)
        iterat.append(0)
        costs.append(self.cost(Y, Aout))
        if verbose:
            print("Cost after", 0, "iterations:", costs[-1])

        for i in range(1, iterations + 1):
            Aout, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                iterat.append(i)
                costs.append(self.cost(Y, Aout))
                if verbose:
                    print("Cost after", i, "iterations:", costs[-1])

        if graph:
            plt.plot(iterat, costs, '-b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Method to save the weights of the NN."""
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Method to load the weights of the NN."""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None


def sigmoid(z):
    """sigmoid function of np array."""
    sig = 1 / (1 + np.exp(-z))

    return sig
