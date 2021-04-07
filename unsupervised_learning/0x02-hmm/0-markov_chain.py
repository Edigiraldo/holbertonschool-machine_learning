#!/usr/bin/env python3
"""Function markov_chain."""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain
    being in a particular state after a specified number of
    iterations.

    - P is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix.
        - P[i, j] is the probability of transitioning from state
          i to state j.
        - n is the number of states in the markov chain.
    - s is a numpy.ndarray of shape (1, n) representing the probability
      of starting in each state.
    - t is the number of iterations that the markov chain has been through.

    Returns: a numpy.ndarray of shape (1, n) representing the probability
    of being in a specific state after t iterations, or None on failure.
    """
    if (type(P) is not np.ndarray or P.ndim != 2 or
       P.shape[0] != P.shape[1]):
        return None
    n = P.shape[0]

    if (P < 0).any() or (P > 1).any():
        return None

    if (type(s) is not np.ndarray or s.ndim != 2 or
       s.shape != (1, n)):
        return None

    if (type(t) is not int or t < 0):
        return None

    P_t = np.identity(n)
    for i in range(t):
        P_t = P @ P_t

    return s @ P_t
