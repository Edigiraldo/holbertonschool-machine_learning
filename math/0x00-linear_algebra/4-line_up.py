#!/usr/bin/env python3
"""Adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays element by element."""
    if len(arr1) == len(arr2):
        sum = []
        len1 = len(arr1)
        for idx in range(len1):
            sum.append(arr1[idx] + arr2[idx])
        return sum
