#!/usr/bin/env python3
"""Adding two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise."""

    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        sum = []
        for i in range(len(mat1)):
            row_sum = []
            for j in range(len(mat1[0])):
                row_sum.append(mat1[i][j] + mat2[i][j])
            sum.append(row_sum)

        return sum
