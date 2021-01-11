#!/usr/bin/env python3
"""Function normalization_constants."""
import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization
     (standardization) constants of a matrix."""
    mean = X.mean(axis=0)
    stddev = X.std(axis=0)

    nx = X.shape[1]

    for i in range(nx):
        X[:, i] -= mean[i]
        X[:, i] /= stddev[i]

    return mean, stddev
