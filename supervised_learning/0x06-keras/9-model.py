#!/usr/bin/env python3
"""Functions save_model and load_model."""
import tensorflow.keras as K


def save_model(network, filename):
    """Function that saves an entire model."""
    K.models.save_model(model=network,
                        filepath=filename)

def load_model(filename):
    """Function that loads an entire model."""
    return K.models.load_model(filepath=filename)
