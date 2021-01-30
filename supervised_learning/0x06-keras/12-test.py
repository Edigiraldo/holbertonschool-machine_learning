#!/usr/bin/env python3
"""Function test_model."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network."""
    return network.evaluate(x=data,
                            y=labels,
                            verbose=verbose)
