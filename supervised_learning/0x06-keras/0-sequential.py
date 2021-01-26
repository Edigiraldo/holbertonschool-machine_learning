#!/usr/bin/env python3
"""Function build_model."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural
    network with the Keras library."""
    model = K.Sequential()
    n_layers = len(layers)

    ker_reg = K.regularizers.l2(lambtha)

    model.add(K.layers.Dense(units=layers[0],
                             activation=activations[0],
                             kernel_regularizer=ker_reg,
                             input_shape=(nx,)))

    if n_layers != 1:
        model.add(K.layers.Dropout(1 - keep_prob))

    for i in range(1, n_layers):
        ker_reg = K.regularizers.l2(lambtha)
        model.add(K.layers.Dense(units=layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=ker_reg))
        if i != n_layers - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
