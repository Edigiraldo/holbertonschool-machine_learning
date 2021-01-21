#!/usr/bin/env python3
"""Function early_stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that determines if you
    should stop gradient descent early."""
    if (opt_cost - cost <= threshold and
       count >= patience - 1):

        return (True, count + 1)

    if opt_cost - cost > threshold:
        count = -1

    return (False, count + 1)
