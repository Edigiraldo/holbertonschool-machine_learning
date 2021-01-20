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

        weights[w_k] = weights[w_k] - alpha * (lambtha / m) * weights[w_k]
        - alpha * (np.matmul(dz, cache["A{}".format(i - 1)].T) / m)
        weights[b_k] = weights[b_k]
        - alpha * (np.sum(dz, axis=1, keepdims=True) / m)
