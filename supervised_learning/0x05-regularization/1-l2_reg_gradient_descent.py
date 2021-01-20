#!/usr/bin/env python3
"""Function l2_reg_gradient_descent."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural
    network using gradient descent with L2 regularization."""
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

        weights["W" + str(i)] = weights["W" + str(i)] - (alpha * ((
                lambtha / m) * weights["W" + str(i)] + dw))
        weights["b" + str(i)] = weights["b" + str(i)] - (alpha * db)
