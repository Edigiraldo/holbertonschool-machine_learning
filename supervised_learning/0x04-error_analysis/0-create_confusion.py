#!/usr/bin/env python3
"""Function create_confusion_matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix."""
    m = labels.shape[0]
    classes = labels.shape[1]

    confusion = np.zeros((classes, classes))

    real = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)

    for i in range(m):
        confusion[real[i], predicted[i]] += 1

    return confusion
