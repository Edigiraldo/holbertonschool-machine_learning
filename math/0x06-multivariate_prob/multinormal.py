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
