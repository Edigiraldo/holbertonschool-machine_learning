#!/usr/bin/env python3
"""Function correlation."""
import numpy as np


def correlation(C):
    """
    Function that calculates a correlation matrix.

    - C is a numpy.ndarray of shape (d, d) containing a
      covariance matrix.
        - d is the number of dimensions

    - Returns a numpy.ndarray of shape (d, d) containing
      the correlation matrix.
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]

    D = np.sqrt(np.diag(1 / np.diag(C)))

    corr = D @ C @ D

    return corr
