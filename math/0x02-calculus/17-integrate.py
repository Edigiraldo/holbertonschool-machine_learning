#!/usr/bin/env python3
"""Integrating polynomials."""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial."""
    if type(poly) != list or len(poly) == 0 or (type(C) != int and type(C) != float):
        return None

    for ele in poly:
        if type(ele) != int and type(ele) != float:
            return None

    for i in range(len(poly)):
        poly[i] = poly[i] / (i + 1)

    for i in range(len(poly)):
        if poly[i] != 0:
            break
        if i == (len(poly) - 1):
            return [0]

    res = [C] + poly
    for i in range(len(res)):
        if res[i] - int(res[i]) == 0.0:
            res[i] = int(res[i])

    return res
