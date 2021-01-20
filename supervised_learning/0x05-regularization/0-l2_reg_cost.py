#!/usr/bin/env python3
"""Function l2_reg_cost."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a
    neural network with L2 regularization."""
    regul_cost = 0
    for i in range(1, L + 1):
        Wn = weights['W' + str(i)]
        regul_cost += np.sum(Wn ** 2)

    regul_cost *= lambtha / (2 * m)

    return cost + regul_cost
