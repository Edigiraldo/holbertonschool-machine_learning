#!/usr/bin/env python3
"""Function likelyhood."""
import numpy as np


def binomial(k, n, p):
    """
    Probability of getting exactly k successes
    in n independent Bernoulli trials with p
    probability of success.
    """
    bin_coef = np.math.factorial(n)
    bin_coef /= np.math.factorial(k) * np.math.factorial(n - k)

    prob = bin_coef * (p ** k) * ((1 - p) ** (n - k))

    return prob


def likelihood(x, n, P):
    """
    Function that calculates the likelihood of obtaining
    this data given various hypothetical probabilities of
    developing severe side effects.

       -  x is the number of patients that develop severe side effects.
       -  n is the total number of patients observed.
       -  P is a 1D numpy.ndarray containing the various hypothetical
          probabilities of developing severe side effects.

    Returns: a 1D numpy.ndarray containing the likelihood of obtaining
    the data, x and n, for each probability in P, respectively.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if (P < 0).any() or (P > 1).any():
        raise ValueError("All values in P must be in the range [0, 1]")

    p_x_given_n = binomial(x, n, P)

    return p_x_given_n
