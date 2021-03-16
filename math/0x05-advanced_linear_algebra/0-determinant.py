#!/usr/bin/env python3
"""determinant function."""


def aux_det(matrix):
    """Function that computes the determinant of a matrix."""
    rows = len(matrix)
    if rows == 1:
        return matrix[0][0]

    det = 0
    for i in range(rows):
        sub_matrix = [matrix[r][1:] for r in range(rows) if r != i]
        det += (-1) ** (i) * matrix[i][0] * aux_det(sub_matrix)

    return det


def determinant(matrix):
    """Function that calculates the determinant of a matrix."""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    rows = len(matrix)

    # Case 0-dim matrix
    if rows == 1 and type(matrix[0]) is list and len(matrix[0]) == 0:
        return 1

    for ele in matrix:
        if type(ele) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(ele) != rows:
            raise ValueError("matrix must be a square matrix")

    return aux_det(matrix)
