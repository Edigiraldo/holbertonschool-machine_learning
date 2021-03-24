#!/usr/bin/env python3
"""pca function."""
import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset.

    - X is a numpy.ndarray of shape (n, d) where:
        - n is the number of data points.
        - d is the number of dimensions in each point.
        - all dimensions have a mean of 0 across all data points.
    - var is the fraction of the variance that the PCA
      transformation should maintain.

    Returns: the weights matrix, W, that maintains var
             fraction of X‘s original variance.
        - W is a numpy.ndarray of shape (d, nd) where nd
          is the new dimensionality of the transformed X.
    """
    U, S, Vh = np.linalg.svd(X)

    cum_var = [S[0]]

    for i in range(1, len(S)):
        cum_var.append(S[i] + cum_var[-1])

    idx = len(cum_var) - 1
    for i in range(len(cum_var)):
        if cum_var[i] / cum_var[-1] >= var:
            idx = i
            break

    return Vh.T[:, :i + 1]
