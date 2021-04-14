#!/usr/bin/env python3
"""Class GaussianProcess."""
import numpy as np


class GaussianProcess:
    """Class that represents a noiseless 1D Gaussian process."""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        - X_init is a numpy.ndarray of shape (t, 1) representing
          the inputs already sampled with the black-box function.
        - Y_init is a numpy.ndarray of shape (t, 1) representing
          the outputs of the black-box function for each input
          in X_init.
        - t is the number of initial samples.
        - l is the length parameter for the kernel.
        - sigma_f is the standard deviation given to the output
          of the black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def predict(self, X_s):
        """
        Function that predicts the mean and standard deviation
        of points in a Gaussian process.
        """
        X_train = self.X
        Y_train = self.Y
        sigma_y = 0

        K = self.K
        K_s = self.kernel(X_train, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(Y_train)
        mu_s = mu_s.reshape(-1)
        cov_s = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu_s, cov_s

    def update(self, X_new, Y_new):
        """
        Function that updates a Gaussian Process:

        - X_new is a numpy.ndarray of shape (1,) that represents
          the new sample point.
        - Y_new is a numpy.ndarray of shape (1,) that represents
          the new sample function value.
        - Updates the public instance attributes X, Y, and K.
        """
        self.X = np.append(self.X, [X_new], axis=0)
        self.Y = np.append(self.Y, [Y_new], axis=0)
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Isotropic squared exponential kernel.

        - X1: Array of m points (m x d).
        - X2: Array of n points (n x d).

        Returns:
            (m x n) kernel matrix.
        """
        l = self.l
        sigma_f = self.sigma_f
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1) +
                  np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T))

        return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)
