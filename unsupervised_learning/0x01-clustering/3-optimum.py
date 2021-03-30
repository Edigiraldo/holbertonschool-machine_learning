#!/usr/bin/env python3
"""Function optimum_k."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Function that tests for the optimum number of clusters
    by variance.

    - X is a numpy.ndarray of shape (n, d) containing the data set.
    - kmin is a positive integer containing the minimum number
      of clusters to check for (inclusive).
    - kmax is a positive integer containing the maximum number
      of clusters to check for (inclusive).
    - iterations is a positive integer containing the maximum
      number of iterations for K-means.

    Returns: (results, d_vars), or (None, None) on failure.
        - results is a list containing the outputs of K-means
          for each cluster size.
        - d_vars is a list containing the difference in variance
          from the smallest cluster size for each cluster size.
    """
    if (type(kmin) is not int or kmin < 1 or
       type(kmax) is not int or kmax < 1 or kmin >= kmax)
        return None, None

    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        var = variance(X, C)
        if var is None:
            return None, None
        variances.append(var)

    variances = [abs(variances[i] - variances[0])
                 for i in range(len(variances))]

    return results, variances
