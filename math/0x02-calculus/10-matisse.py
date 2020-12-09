#!/usr/bin/env python3
"""Deriving polinomials."""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial."""
    if type(poly) != list:
        return None

    for ele in poly:
        if type(ele) != int and type(ele) != float:
            return None

    for i in range(len(poly)):
        poly[i] = poly[i] * i

    for i in range(len(poly)):
        if poly[i] != 0:
            break
        if i == (len(poly) - 1):
            return [0]

    return poly[1:]
