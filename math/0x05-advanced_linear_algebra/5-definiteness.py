#!/usr/bin/env python3
"""determinant function."""
import numpy as np


def definiteness(matrix):
    """Function that calculates the definiteness of a matrix.

       matrix is a numpy.ndarray of shape (n, n) whose
       definiteness should be calculated.
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if (matrix.ndim != 2 or
       matrix.shape[0] != matrix.shape[1] or
       matrix.shape[1] == 0 or
       (np.matrix.conjugate(matrix.T) != matrix).all()):

        return None

    eig = np.linalg.eig(matrix)[0]

    if (eig > 0).all():
        return "Positive definite"
    if (eig < 0).all():
        return "Negative definite"
    if (eig >= 0).all():
        return "Positive semi-definite"
    if (eig <= 0).all():
        return "Negative semi-definite"
    else:
        return "Indefinite"
