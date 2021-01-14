#!/usr/bin/env python3
"""Function batch_norm."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output
    of a neural network using batch normalization."""
    Z_norm = (Z - np.mean(Z, axis=0, keepdims=True))
    Z_norm /= np.sqrt(np.var(Z, axis=0, keepdims=True) + epsilon)

    Z_batch_norm = gamma * Z_norm + beta

    return Z_batch_norm
