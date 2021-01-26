#!/usr/bin/env python3
"""Function train_model."""
import tensorflow.keras as K

def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """Function that trains a model using
    mini-batch gradient descent."""
    hist = network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)

    return hist
