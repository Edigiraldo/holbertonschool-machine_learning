#!/usr/bin/env python3
"""Normal distribution."""


class Normal:
    """Class Normal that represents a normal distribution."""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = standardev(data, self.mean)


def standardev(data, mean):
    """Returns the population standard deviation."""

    dev = 0
    for ele in data:
        dev += (ele - mean) ** 2

    dev /= len(data)

    return dev ** 0.5
