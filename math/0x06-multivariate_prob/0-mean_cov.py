#!/usr/bin/env python3
"""mean_cov function."""
import numpy as np


def mean_cov(X):
    """
        Function that calculates the mean and covariance of a data set.

        - X is a numpy.ndarray of shape (n, d) containing the data set:
            - n is the number of data points
            - d is the number of dimensions in each data point
        Returns: mean, cov:
            - mean is a numpy.ndarray of shape (1, d) containing
              the mean of the data set.
            - cov is a numpy.ndarray of shape (d, d) containing
              the covariance matrix of the data set.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n = X.shape[0]

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_norm = X - mean

    cov = (X_norm.T @ X_norm) / (n - 1)

    return mean, cov
