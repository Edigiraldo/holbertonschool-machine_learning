#!/usr/bin/env python3
"""Concatenating two 2D matrices along an specific axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis."""

    concat = []

    mat1_cpy = []
    for row in mat1:
        mat1_cpy.append(row.copy())
    concat.extend(mat1_cpy)

    mat2_cpy = []
    for row in mat2:
        mat2_cpy.append(row.copy())

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        concat.extend(mat2_cpy)
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat2)):
            concat[i].extend(mat2_cpy[i])

    return concat
