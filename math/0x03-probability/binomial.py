#!/usr/bin/env python3
"""Binomial distribution."""


class Binomial:
    """Class Binomial that represents a binomial distribution."""
    def __init__(self, data=None, n=1, p=0.5):
        """Constructor method."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)

            self.p = float(1 - (variance(data) / mean))
            self.n = int(mean / self.p)
            self.p = float(mean/self.n)

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of “successes”."""
        if k < 0:
            return 0
        k = int(k)

        pmf = binomial_coef(self.n, k)
        pmf *= (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF
        for a given number of “successes”."""
        if k < 0:
            return 0
        k = int(k)

        cdf = 0
        for x in range(0, k + 1):
            cdf += self.pmf(x)

        return cdf


def variance(data):
    """Computes the variance of a given distribution."""
    mean = sum(data) / len(data)
    var = 0

    for ele in data:
        var += (ele - mean) ** 2

    return var / len(data)


def factorial(n):
    """Calculates n!"""

    if type(n) != int or n < 0:
        return None

    fact = 1

    for i in range(n, 0, -1):
        fact *= i

    return fact


def binomial_coef(n, k):
    """Computes the binomial coeficient. n > k."""
    return factorial(n) / (factorial(k) * factorial(n - k))
