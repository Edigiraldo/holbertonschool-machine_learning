#!/usr/bin/env python3
"""Function dropout_gradient_descent."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network
    with Dropout regularization using gradient descent."""
    m = Y.shape[1]
    d_a = 0

    for i in reversed(range(L)):
        if i == L - 1:
            d_z = cache['A' + str(L)] - Y

        else:
            d_z = (d_a * (1 - cache['A' + str(i + 1)] **
                          2) * cache['D' + str(i + 1)]) / keep_prob

        d_w = np.matmul(d_z, cache['A' + str(i)].T) / m
        d_b = np.sum(d_z, axis=1, keepdims=True) / m
        d_a = np.matmul(weights['W' + str(i + 1)].T, d_z)

        weights['W' + str(i + 1)] = weights['W' + str(i + 1)] - alpha * d_w
        weights['b' + str(i + 1)] = weights['b' + str(i + 1)] - alpha * d_b
