#!/usr/bin/env python3
"""numpy arrays element-wise operations."""


def np_elementwise(mat1, mat2):
    """function that performs element-wise addition,
    subtraction, multiplication, and division"""

    a = mat1 + mat2
    s = mat1 - mat2
    m = mat1 * mat2
    d = mat1 / mat2

    return a, s, m, d
