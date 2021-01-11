#!/usr/bin/env python3
"""Function normalize."""
import numpy as np


def normalize(X, m, s):
    """Function that normalizes (standardizes) a matrix."""
    nx = X.shape[1]

    for i in range(nx):
        X[:, i] -= m[i]
        X[:, i] /= s[i]

    return X
