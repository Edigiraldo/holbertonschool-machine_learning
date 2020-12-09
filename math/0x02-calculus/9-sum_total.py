#!/usr/bin/env python3
"""Sum of squares."""


def summation_i_squared(n):
    """Calculates sum of squares from i = 1 to i = n."""
    if n > 0:
        sum = (n * (n + 1) * (2 * n + 1)) / 6
        return sum
