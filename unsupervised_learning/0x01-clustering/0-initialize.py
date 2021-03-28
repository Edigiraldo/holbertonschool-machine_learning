#!/usr/bin/env python3
"""Function initialize."""
import numpy as np


def initialize(X, k):
    """
    Function that initializes cluster centroids for K-means.

    - X is a numpy.ndarray of shape (n, d) containing the
      dataset that will be used for K-means clustering.
        - n is the number of data points.
        - d is the number of dimensions for each data point.
    - k is a positive integer containing the number of clusters.

    Returns: a numpy.ndarray of shape (k, d) containing the
    initialized centroids for each cluster, or None on failure.
    """
    if (type(X) != np.ndarray or X.ndim != 2 or type(k) is not int
       or k < 1):
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)
    d = X.shape[1]
    init = np.random.uniform(low=low, high=high, size=(k, d))

    return init
