#!/usr/bin/env python3
"""Poisson distribution."""

e = 2.7182818285


class Poisson:
    """Class Poisson that represents a poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.data = data
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of “successes”."""
        if k < 0:
            return 0
        k = int(k)

        return ((self.lambtha ** k) * (e ** (-self.lambtha))) / (factorial(k))

    def cdf(self, k):
        """Calculates the value of the CDF
        for a given number of “successes”."""
        if k < 0:
            return 0
        k = int(k)

        sum = 0
        for i in range(0, k+1):
            sum += (self.lambtha ** i) / factorial(i)

        return (e ** (-self.lambtha)) * sum


def factorial(n):
    """Calculates n!"""

    if type(n) != int or n < 0:
        return None

    fact = 1

    for i in range(n, 0, -1):
        fact *= i

    return fact
