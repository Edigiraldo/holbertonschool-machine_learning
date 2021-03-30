#!/usr/bin/env python3
"""Function pdf."""
import numpy as np


def pdf(X, m, S):
    """
    Function that calculates the probability
    density function of a Gaussian distribution.

    - X is a numpy.ndarray of shape (n, d) containing the data
      points whose PDF should be evaluated.
    - m is a numpy.ndarray of shape (d,) containing the mean
      of the distribution.
    - S is a numpy.ndarray of shape (d, d) containing the
      covariance of the distribution.

    Returns: P, or None on failure
        - P is a numpy.ndarray of shape (n,) containing the PDF
          values for each data point.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None

    n, d = X.shape

    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    den = np.sqrt((2 * np.pi) ** d * S_det)
    expo = (X - m) @ S_inv @ (X - m).T

    P = np.exp(- expo / 2) / den
    P = np.einsum('ii->i', P)
    P[P < 1e-300] = 1e-300

    return P
