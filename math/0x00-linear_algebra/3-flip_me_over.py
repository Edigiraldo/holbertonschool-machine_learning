#!/usr/bin/env python3
"""Transpose a matrix."""


def matrix_transpose(matrix):
    """Returns the transpose of an 2D matrix."""
    transpose = []
    for col in range(len(matrix[0])):
        column = []
        for row in range(len(matrix)):
            column.append(matrix[row][col])
        transpose.append(column)

    return transpose
