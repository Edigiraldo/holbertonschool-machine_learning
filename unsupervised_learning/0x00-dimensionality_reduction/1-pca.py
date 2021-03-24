#!/usr/bin/env python3
"""pca function."""
import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset.

    - X is a numpy.ndarray of shape (n, d) where:
        - n is the number of data points
        - d is the number of dimensions in each point
    - ndim is the new dimensionality of the transformed X

    Returns: T, a numpy.ndarray of shape (n, ndim)
             containing the transformed version of X.
    X_m = X - np.mean(X, axis=0)
    U, S, Vh = np.linalg.svd(X_m)

    T = U[:, :ndim] @ np.diag(S)[:ndim, :ndim]
    # T = X_m @ Vh.T[:, :ndim]
    """
    X = X - np.mean(X, axis=0)
    _, S, V = np.linalg.svd(X)
    idx = S.argsort()[::-1]
    eig_val = S[idx]
    eig_vect = V.T
    eig_vect = eig_vect[:, idx]

    w = eig_vect[:, 0: ndim]
    T = np.matmul(X, w)

    return T
