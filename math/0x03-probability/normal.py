#!/usr/bin/env python3
"""Normal distribution."""
pi = 3.1415926536
e = 2.7182818285


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

    def z_score(self, x):
        """Calculates the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value."""
        y = (-0.5) * ((x - self.mean) / self.stddev) ** 2
        pdf = (e ** y) / (self.stddev * (2 * pi) ** 0.5)

        return pdf


def standardev(data, mean):
    """Returns the population standard deviation."""

    dev = 0
    for ele in data:
        dev += (ele - mean) ** 2

    dev /= len(data)

    return dev ** 0.5
