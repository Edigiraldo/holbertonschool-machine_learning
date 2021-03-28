#!/usr/bin/env python3
"""Function variance."""
import numpy as np


def distances(a, b):
    """
    Function that computes the euclidian distance
    from all points in a, whit every other in b.

    - a ndarray of shape(n, d).
    - b ndarray of shape(m, d).
        - n and m are the number of points.
        - d is the dimention of each point.

    Returns: (n, m) ndarray with distances.
    """
    b = b[np.newaxis, :]
    a = a[:, np.newaxis, :]

    diff = a - b
    dist = np.linalg.norm(diff, axis=-1, keepdims=False)

    return dist


def variance(X, C):
    """
    Function that calculates the total intra-cluster
    variance for a data set.

    - X is a numpy.ndarray of shape (n, d) containing the data set.
    - C is a numpy.ndarray of shape (k, d) containing the centroid
      means for each cluster.

    Returns: var, or None on failure.
    """
    if (type(X) != np.ndarray or X.ndim != 2 or
       type(C) != np.ndarray or X.ndim != 2):
        return None

    n = X.shape[0]
    d = X.shape[1]
    k = C.shape[1]
    if (n == 0 or d == 0 or C.shape[1] != d or k == 0):
        return None

    dist = distances(X, C)
    clss = np.argmin(dist, axis=1).reshape(-1)
    dist_clss = np.ones(dist.shape, dtype=bool)
    dist_clss[np.arange(dist.shape[0]), clss] = False
    dist[dist_clss] = 0
    square = dist ** 2
    var = np.sum(square)

    return var
