#!/usr/bin/env python3
"""One-Hot Encode."""
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric
    label vector into a one-hot matrix."""
    if (type(Y) != np.ndarray or Y.shape[0] != Y.size
       or Y.size == 0 or type(classes) != int
       or classes <= 0):

        return None

    ohe = np.zeros((classes, Y.shape[0]))
    ohe[Y, np.arange(classes)] = 1

    return ohe
