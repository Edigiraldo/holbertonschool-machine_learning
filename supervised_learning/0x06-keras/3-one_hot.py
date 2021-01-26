#!/usr/bin/env python3
"""Function one_hot."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Function that converts a label
    vector into a one-hot matrix."""
    ohe = K.utils.to_categorical(y=labels,
                                 num_classes=classes)

    return ohe
