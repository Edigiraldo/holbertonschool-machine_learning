#!/usr/bin/env python3
"""Function cofactor."""


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


def minor(matrix):
    """Function that calculates the minor matrix of a matrix."""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    rows = len(matrix)

    for ele in matrix:
        if type(ele) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(ele) != rows:
            raise ValueError("matrix must be a non-empty square matrix")

    if rows == 1:
        return [[1]]

    minors = [[0 for j in range(rows)] for i in range(rows)]

    for i in range(rows):
        for j in range(rows):
            sub_matrix = [[matrix[r][c] for c in range(rows) if c != j]
                          for r in range(rows) if r != i]
            minors[i][j] = aux_det(sub_matrix)

    return minors


def cofactor(matrix):
    """Function that calculates the cofactor matrix of a matrix."""
    minors = minor(matrix)
    rows = len(minors)
    cols = len(minors[0])
    for i in range(rows):
        for j in range(cols):
            minors[i][j] *= (-1) ** (i + j)

    return minors
