#!/usr/bin/env python3
"""Multiplying 2D matrices."""


def mat_mul(mat1, mat2):
    """function that performs matrix multiplication."""

    if len(mat1[0]) != len(mat2):
        return None

    mult = zeros_mat(len(mat1), len(mat2[0]))

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):

            # Getting column of second matrix
            colj = []
            for k in range(len(mat2)):
                colj.append(mat2[k][j])

            mult[i][j] = dot_product(mat1[i], colj)

    return mult


def dot_product(vec1, vec2):
    """Returns the dot product of two vectors."""

    if len(vec1) != len(vec2):
        print(vec1, vec2)
        return None

    dot = 0
    for i in range(len(vec1)):
        dot += vec1[i] * vec2[i]

    return dot


def zeros_mat(rows, cols):
    """Creates a matrix of zeros and shape (rows, cols)."""

    zeros = []
    for i in range(rows):
        row = cols * [0]
        zeros.append(row)

    return zeros
