#!/usr/bin/env python3
"""Class BayesianOptimization."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Class that performs Bayesian optimization on a
    noiseless 1D Gaussian process."""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
            - f is the black-box function to be optimized
            - X_init is a numpy.ndarray of shape (t, 1) representing
              the inputs already sampled with the black-box function.
            - Y_init is a numpy.ndarray of shape (t, 1) representing
              the outputs of the black-box function for each input
              in X_init.
            - t is the number of initial samples.
            - bounds is a tuple of (min, max) representing the bounds
              of the space in which to look for the optimal point.
            - ac_samples is the number of samples that should be
              analyzed during acquisition.
            - l is the length parameter for the kernel.
            - sigma_f is the standard deviation given to the output
              of the black-box function.
            - xsi is the exploration-exploitation factor for acquisition.
            - minimize is a bool determining whether optimization
              should be performed for minimization (True) or
              maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        b_min, b_max = bounds
        self.X_s = np.linspace(b_min, b_max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
            Function that calculates the next best sample location.

            - X_next is a numpy.ndarray of shape (1,) representing
              the next best sample point.
            - EI is a numpy.ndarray of shape (ac_samples,) containing
              the expected improvement of each potential sample.
        """
        # Most likely function.
        mu, sigma = self.gp.predict(self.X_s)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        if self.minimize:
            min_val = np.min(self.gp.Y)
            # f(x-) - u(x) - xsi
            num = min_val - mu - self.xsi
        else:
            max_val = np.max(self.gp.Y)
            # u(x) - f(x+) - xsi
            num = mu - max_val - self.xsi

        Z = num.astype(float) / sigma.astype(float)
        # zero division handling
        Z[Z == np.inf] = 0

        cdf_Z = norm.cdf(Z)
        pdf_Z = norm.pdf(Z)

        EI = num * cdf_Z + sigma * pdf_Z
        # best point to explore
        best_X = self.X_s[np.argmax(EI)]

        EI = EI.reshape(-1)

        return best_X, EI

    def optimize(self, iterations=100):
        """
        Function that optimizes the black-box function.

        - iterations is the maximum number of iterations to perform.
        """
        GP = self.gp
        for i in range(iterations):
            X_new, EI = self.acquisition()
            X = self.gp.X
            if (X_new == X).any():
                break

            Y_new = self.f(X_new)
            GP.update(X_new, Y_new)

        if self.minimize:
            idx = np.argmin(GP.Y)
        else:
            idx = np.argmax(GP.Y)

        return GP.X[idx], GP.Y[idx]
