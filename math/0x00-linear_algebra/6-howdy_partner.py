#!/usr/bin/env python3
"""Concatenating lists."""


def cat_arrays(arr1, arr2):
    """function that concatenates two arrays."""

    concat = []
    concat.extend(arr1)
    concat.extend(arr2)

    return concat
