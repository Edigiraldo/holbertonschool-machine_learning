#!/usr/bin/env python3
"""Function regular."""
import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities
    of a regular markov chain.

    - P is a square 2D numpy.ndarray of shape (n, n)
      representing the transition matrix.
        - P[i, j] is the probability of transitioning from
          state i to state j.
        - n is the number of states in the markov chain.

    Returns: a numpy.ndarray of shape (1, n) containing the
    steady state probabilities, or None on failure.
    """
    if (type(P) is not np.ndarray or P.ndim != 2 or
       P.shape[0] != P.shape[1]):
        return None
    n = P.shape[0]

    if (P < 0).any() or (P > 1).any():
        return None

    # s@P = s -> P.T @ s.T = s.T
    evals, evects = np.linalg.eig(P.T)
    eval_1 = np.argmin(np.abs(evals - 1))

    if not np.isclose(evals[eval_1],  1):
        return None

    if np.sum(np.isclose(evals, 1)) != 1:
        return None

    ss = evects[:, eval_1].reshape(1, -1)

    ss = ss / np.sum(ss)

    return ss
