#!/usr/bin/env python3
"""Function update_variables_momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function that updates a variable using the
    gradient descent with momentum optimization algorithm."""
    momentum = beta1 * v + (1 - beta1) * grad
    
    return var - alpha * momentum, momentum
