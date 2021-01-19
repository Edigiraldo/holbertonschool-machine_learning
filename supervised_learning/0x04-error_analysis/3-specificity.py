#!/usr/bin/env python3
"""Function specificity."""
import numpy as np


def specificity(confusion):
    """Function that calculates the specificity
    for each class in a confusion matrix."""
    classes = confusion.shape[0]
    sum_all = np.sum(confusion)
    tn = np.zeros(classes) + sum_all
    for i in range(classes):
        tn[i] -= np.sum(confusion[i, :])
        tn[i] -= np.sum(confusion[:, i])
        tn[i] += confusion[i, i]

    fp = np.sum(confusion, axis=0)
    for i in range(classes):
        fp[i] -= confusion[i, i]

    return tn / (tn + fp)
