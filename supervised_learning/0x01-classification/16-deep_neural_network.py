#!/usr/bin/env python3
"""DeepNeuralNetwork class."""
import numpy as np


class DeepNeuralNetwork:
    """class that defines a deep neural network
       performing binary classification."""
    def __init__(self, nx, layers):
        """Class constructor."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # A dictionary to hold all intermediary values of the network.
        self.cache = {}
        # A dictionary to hold all weights and biased of the network.
        self.weights = {}
        # The number of layers in the neural network.
        self.L = len(layers)

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

            self.weights[nameW] = np.random.randn(neurons,
                                                  weights)*np.sqrt(2/weights)
            self.weights[nameb] = np.zeros((neurons, 1))
