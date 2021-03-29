#!/usr/bin/env python3
"""Function initialize."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Function that initializes variables for a Gaussian Mixture Model.

    - X is a numpy.ndarray of shape (n, d) containing the data set.
    - k is a positive integer containing the number of clusters.

    Returns: pi, m, S, or None, None, None on failure.
        - pi is a numpy.ndarray of shape (k,) containing the priors
          for each cluster, initialized evenly.
        - m is a numpy.ndarray of shape (k, d) containing the centroid
          means for each cluster, initialized with K-means.
        - S is a numpy.ndarray of shape (k, d, d) containing the
          covariance matrices for each cluster, initialized as
          identity matrices.
    """
    if (type(X) != np.ndarray or X.ndim != 2 or type(k) is not int
       or k < 1 or X.shape[0] < k):
        return None, None, None

    n = X.shape[0]
    kmeans(X, k, iterations=1000)
    idxs = np.random.choice(n, size=k, replace=False)

    pi = np.ones(k) / k
    m = X[idxs].copy()

    m_d = m[:, np.newaxis, :]
    X_d = X[np.newaxis, :, :]

    norm = X_d - m_d
    normT = np.transpose(norm, (0, 2, 1))
    S = (normT @ norm) / (n - 1)

    return pi, m, S
