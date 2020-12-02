#!/usr/bin/env python3
"""Getting a matrix shape."""


def matrix_shape(matrix):
    """Returns the shape of matrix."""
    shape_list = []
    shape(matrix, shape_list)

    return shape_list


def shape(matrix, shape_list):
    """shape of the matrix"""
    if type(matrix) == list and len(matrix) > 0:
        shape_list.append(len(matrix))
        shape(matrix[0], shape_list)
