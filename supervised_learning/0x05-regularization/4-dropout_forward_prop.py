#!/usr/bin/env python3
"""Function dropout_forward_prop."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward
    propagation using Dropout."""
    n_layers = L
    cache = {}
    cache['A0'] = X

    for i in range(n_layers):
        inputs = cache['A' + str(i)]
        weight = weights['W' + str(i + 1)]
        bias = weights['b' + str(i + 1)]

        if i == n_layers - 1:
            cache['A' + str(i + 1)] = softmax(weight @ inputs + bias)
        else:
            A = np.tanh(weight @ inputs + bias) / keep_prob
            drop_mask = np.random.rand(A.shape[0], A.shape[1])
            drop_mask = np.where(drop_mask < keep_prob, 1, 0)
            A = A * drop_mask
            cache['A' + str(i + 1)] = A
            cache['D' + str(i + 1)] = drop_mask

    return cache


def softmax(x):
        """Softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
