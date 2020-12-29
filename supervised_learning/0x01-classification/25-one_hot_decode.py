#!/usr/bin python3
"""One hot decode."""
import numpy as np


def one_hot_decode(one_hot):
    """Function that converts a one-hot
    matrix into a vector of labels."""
    if (type(one_hot) != np.ndarray or one_hot.ndim != 2
       or one_hot.shape[0] == 0 or one_hot.shape[1] == 0):
        return None

    return np.argmax(one_hot, axis=0)
