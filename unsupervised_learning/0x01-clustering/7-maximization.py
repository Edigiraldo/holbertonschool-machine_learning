#!/usr/bin/env python3
"""Function maximization."""
import numpy as np


def maximization(X, g):
    """
    Function that calculates the maximization step in the
    EM algorithm for a GMM.

    - X is a numpy.ndarray of shape (n, d) containing the data set.
    - g is a numpy.ndarray of shape (k, n) containing the posterior
      probabilities for each data point in each cluster.

    Returns: pi, m, S, or None, None, None on failure.
        - pi is a numpy.ndarray of shape (k,) containing the updated
          priors for each cluster.
        - m is a numpy.ndarray of shape (k, d) containing the updated
          centroid means for each cluster.
        - S is a numpy.ndarray of shape (k, d, d) containing the updated
          covariance matrices for each cluster.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    n, d = X.shape

    if (type(g) is not np.ndarray or g.ndim != 2 or
       g.shape[1] != n):
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), 1).all():
        return (None, None, None)

    k = g.shape[0]

    pi = np.sum(g, axis=1) / n
    num = g[:, :, np.newaxis] * X[np.newaxis, :, :]  # ... = (k, n, d)
    m = np.sum(num, axis=1) / np.sum(g, axis=1).reshape(k, 1)  # (k, d)/ (k, 1)

    g_0n = g[0].reshape(n, 1)  # cluster 0
    m_0d = m[0].reshape(1, d)  # mean cluster 0
    S_0 = g_0n * (X - m_0d) ** 2  # (n, d) = (n, 1) * (n, d)
    S_0 = np.cov(S_0.T)  # cov matrix cluster 0 -> (d, d)
    S = S_0[np.newaxis, :]
    for i in range(1, k):
        g_in = g[i].reshape(n, 1)  # cluster i
        m_id = m[i].reshape(1, d)  # mean cluster i
        S_i = g_in * (X - m_id) ** 2  # (n, d) = (n, 1) * (n, d)
        S_i = np.cov(S_i.T)  # cov matrix cluster i -> (d, d)
        S_i = S_i[np.newaxis, :]
        np.concat((S, S_i), axis=0)

    return pi, m, S
