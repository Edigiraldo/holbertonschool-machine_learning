#!/usr/bin/env python3
"""Function posterior."""
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


def intersection(x, n, P, Pr):
    """
    Function that calculates the intersection of obtaining
    this data with the various hypothetical probabilities.

        - x is the number of patients that develop severe side effects.
        - n is the total number of patients observed.
        - P is a 1D numpy.ndarray containing the various hypothetical
          probabilities of developing severe side effects.
        - Pr is a 1D numpy.ndarray containing the prior beliefs of P.

    Returns: a 1D numpy.ndarray containing the intersection of obtaining
    x and n with each probability in P, respectively.
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
    if type(Pr) is not np.ndarray or Pr.ndim != 1:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if Pr.shape[0] != P.shape[0]:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if (P < 0).any() or (P > 1).any():
        raise ValueError("All values in P must be in the range [0, 1]")
    if (Pr < 0).any() or (Pr > 1).any():
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    return Pr * likelihood(x, n, P)


def marginal(x, n, P, Pr):
    """
    Function that calculates the marginal probability of obtaining the data.

        - x is the number of patients that develop severe side effects.
        - n is the total number of patients observed.
        - P is a 1D numpy.ndarray containing the various hypothetical
          probabilities of patients developing severe side effects.
        - Pr is a 1D numpy.ndarray containing the prior beliefs about P.

    Returns: the marginal probability of obtaining x and n.
    """
    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """
    Function that calculates the posterior probability for the various
    hypothetical probabilities of developing severe side effects given
    the data.

        - x is the number of patients that develop severe side effects.
        - n is the total number of patients observed.
        - P is a 1D numpy.ndarray containing the various hypothetical
          probabilities of developing severe side effects.
        - Pr is a 1D numpy.ndarray containing the prior beliefs of P.

    Returns: the posterior probability of each probability in P
    given x and n, respectively.

    1.P(x|n/P)
    2.-------
    3.P(x|n)
    P(P) = 1/P.shape[0]
    4.P(P / x|n)
    """
    P_xn = marginal(x, n, P, Pr)
    P_xn_P = likelihood(x, n, P)
    P_P = 1 / P.shape[0]

    return (P_xn_P * P_P) / P_xn