#!/usr/bin/env python3
"""One-Hot Encode."""
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric
    label vector into a one-hot matrix."""
    if (not isinstance(Y, np.ndarray) or len(Y) == 0 or
            not isinstance(classes, int) or classes <= np.amax(Y)):
        return None

    return np.eye(classes)[Y].T
