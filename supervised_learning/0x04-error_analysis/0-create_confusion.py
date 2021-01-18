#!/usr/bin/env python3
"""Function create_confusion_matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix."""
    m = labels.shape[0]
    confusion = np.zeros((m, m))

    real = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)

    confusion[real, predicted] += 1

    return confusion
