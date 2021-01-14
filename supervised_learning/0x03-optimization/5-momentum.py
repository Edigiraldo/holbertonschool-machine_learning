#!/usr/bin/env python3
"""Funciton update_variables_momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function that updates a variable using the
    gradient descent with momentum optimization algorithm."""
    momentum = beta1 * v + alpha * grad
    return var - momentum, momentum
