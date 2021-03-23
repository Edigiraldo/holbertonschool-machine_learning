#!/usr/bin/env python3
"""Class MultiNormal."""
import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution."""
    def __init__(self, data):
        """
        Class constructor.

        - data is a numpy.ndarray of shape (d, n) containing
          the data set:
            - n is the number of data points.
            - d is the number of dimensions in each data point.
        """
        if type(data) is not np.ndarray or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_norm = (data - self.mean).T

        self.cov = (data_norm.T @ data_norm) / (n - 1)

    def pdf(self, x):
        """
        - x is a numpy.ndarray of shape (d, 1) containing
          the data point whose PDF should be calculated.
            - d is the number of dimensions of the Multinomial instance.

        Returns the value of the PDF.
        """
        d = self.cov.shape[0]
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if x.ndim != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({0}, 1)".format(d))

        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)
        x_norm = x - self.mean

        z = -0.5 * (x_norm.T @ cov_inv @ x_norm)
        pdf = np.exp(z)[0][0] / np.sqrt(((2 * np.pi) ** d) * cov_det)

        return pdf
