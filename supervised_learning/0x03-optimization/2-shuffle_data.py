#!/usr/bin/env python3
"""Function def shuffle_data(X, Y)."""
import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the data
    points in two matrices the same way"""
    m = X.shape[0]
    rows_perm = np.random.permutation(m)

    X = X.copy()
    Y = Y.copy()

    X = X[rows_perm, :]
    Y = Y[rows_perm, :]

    return X, Y
