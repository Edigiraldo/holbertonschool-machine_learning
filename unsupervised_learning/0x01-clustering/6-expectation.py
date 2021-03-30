#!/usr/bin/env python3
"""Function expectation."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Function that calculates the expectation step
    in the EM algorithm for a GMM.

    - X is a numpy.ndarray of shape (n, d) containing the data set
    - pi is a numpy.ndarray of shape (k,) containing the priors
      for each cluster.
    - m is a numpy.ndarray of shape (k, d) containing the centroid
      means for each cluster.
    - S is a numpy.ndarray of shape (k, d, d) containing the
      covariance matrices for each cluster.

    Returns: g, l, or None, None on failure.
        - g is a numpy.ndarray of shape (k, n) containing the
          posterior probabilities for each data point in each cluster.
        - l is the total log likelihood.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    n, d = X.shape

    if (type(m) is not np.ndarray or m.ndim != 2 or
       m.shape[1] != d):
        return None, None
    k = m.shape[0]

    if (type(S) != np.ndarray or S.ndim != 3 or
       S.shape[0] != k or S.shape[1] != d or
       S.shape[2] != d):
        return None, None

    if (type(pi) is not np.ndarray or pi.ndim != 1 or
       pi.shape[0] != k):
        return None, None

    P_0n = pdf(X, m[0], S[0]).reshape(1, -1)
    for i in range(k):
        P_kn = pdf(X, m[i], S[i]).reshape(1, -1)
        P = np.concatenate((P, P_kn), axis=0)
    pi = pi.reshape(k, 1)

    gama_kn = pi * P_kn
    gama_kn = gama_kn / np.sum(gama_kn, axis=0, keepdims=True)

    ll = np.sum(np.log(g.sum(axis=0)))

    return gama_kn, ll
