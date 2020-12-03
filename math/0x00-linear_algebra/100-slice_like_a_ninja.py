#!/usr/bin/env python3
"""Slicing along a specific axis."""


def np_slice(matrix, axes={}):
    """function that slices a matrix along a specific axes."""

    slc = [slice(None)] * len(matrix.shape)

    for axis in axes.keys():
        ax = int(axis)
        slc[ax] = slice(*axes[axis])

    return matrix[tuple(slc)]
