#!/usr/bin/env python3
"""Function precision."""
import numpy as np


def precision(confusion):
    """Function that calculates the precision
    for each class in a confusion matrix."""
    predict_true = np.sum(confusion, axis=0)
    classes = confusion.shape[0]
    tp = np.zeros(classes)

    for i in range(classes):
        tp[i] = confusion[i, i]

    return tp / predict_true
