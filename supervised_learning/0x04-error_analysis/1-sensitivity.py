#!/usr/bin/env python3
"""Function sensitivity."""
import numpy as np


def sensitivity(confusion):
    """Function that calculates the sensitivity
    for each class in a confusion matrix."""
    classes = confusion.shape[0]
    tp = np.zeros(classes)  # true positives
    total = np.sum(confusion, axis=1)
    for i in range(classes):
        tp[i] = confusion[i, i]

    return tp / total
