#!/usr/bin/env python3
"""Monte Carlo Policy Gradient."""
import numpy as np


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix.
    """
    xw = np.dot(matrix, weight)
    e = np.exp(xw)

    soft = e / e.sum()

    return soft    
