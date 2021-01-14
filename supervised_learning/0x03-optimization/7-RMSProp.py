#!/usr/bin/env python3
"""Function update_variables_RMSProp."""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Funciton that updates a variable using
    the RMSProp optimization algorithm."""
    S = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / ((S + epsilon) ** 0.5)

    return var, S
