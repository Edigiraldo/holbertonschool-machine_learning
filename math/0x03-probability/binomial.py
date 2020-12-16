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


def variance(data):
    """Computes the variance of a given distribution."""
    mean = sum(data) / len(data)
    var = 0

    for ele in data:
        var += (ele - mean) ** 2

    return var / len(data)
