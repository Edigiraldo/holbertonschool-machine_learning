#!/usr/bin/env python3
"""Function absorbing."""
import numpy as np


def absorbing(P):
    """
    Function that determines if a markov chain is absorbing.

    - P is a is a square 2D numpy.ndarray of shape (n, n)
      representing the standard transition matrix.
        - P[i, j] is the probability of transitioning from
          state i to state j.
        - n is the number of states in the markov chain.
    Returns: True if it is absorbing, or False on failure.
    """
    if (type(P) is not np.ndarray or P.ndim != 2 or
       P.shape[0] != P.shape[1]):
        return False
    n = P.shape[0]

    if (P < 0).any() or (P > 1).any():
        return False

    if not np.allclose(np.sum(P, axis=1), 1):
        return False

    ones_diag = np.where(np.diag(P) == 1)[0]
    if ones_diag.ndim != 1 or ones_diag.shape[0] == 0:
        return False
    if ones_diag.ndim == n:
        return True
    P_cp = P.copy()
    Q = np.delete(np.delete(P_cp, ones_diag, axis=0), ones_diag, axis=1)
    Id = np.identity(Q.shape[0])

    F_exists = (np.linalg.det(Id - Q) != 0)
    if F_exists:
        return True
    else:
        return False
