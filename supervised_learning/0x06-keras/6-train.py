#!/usr/bin/env python3
"""Function train_model."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Function that trains a model using mini-batch gradient
    descent, to analyze validation data, and to train the model
    using early stopping."""
    es = None
    if validation_data is not None and early_stopping:
        es = K.callbacks.EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       verbose=verbose,
                                       patience=patience)

    hist = network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=[es])

    return hist
