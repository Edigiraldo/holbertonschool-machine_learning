#!/usr/bin/env python3
"""Function dropout_gradient_descent."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network
    with Dropout regularization using gradient descent."""
    m = Y.shape[1]
    weights_c = weights.copy()
    for i in reversed(range(1, L + 1)):
        w_k = "W{}".format(i)
        b_k = "b{}".format(i)
        A = cache['A{}'.format(i)]
        if i == L:
            dz = A - Y
        else:
            dz = (1 - A ** 2) * np.matmul(
                        weights_c["W{}".format(i + 1)].T, dz)

        dw = (dz @ cache["A" + str(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights["W" + str(i)] = weights["W" + str(i)] - (alpha * dw / keep_prob)
        weights["b" + str(i)] = weights["b" + str(i)] - (alpha * db / keep_prob)
