#!/usr/bin/env python3
"""function moving_average."""


def moving_average(data, beta):
    """Function that calculates the weighted
    moving average of a data set."""
    v1 = 0
    mv_avrg = []
    for i in range(0, len(data)):
        v1 = beta * v1 + (1 - beta) * data[i]
        v2 = v1/(1 - beta ** (i + 1))
        mv_avrg.append(v2)

    return mv_avrg
