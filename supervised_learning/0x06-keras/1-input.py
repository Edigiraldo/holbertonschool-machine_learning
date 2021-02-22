#!/usr/bin/env python3
"""Function build_model."""
#import tensorflow.keras as K
from tensorflow import keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural
    network with the Keras library."""
    inputs = K.Input(shape=(nx,))
    n_layers = len(layers)

    out = inputs

    for i in range(0, n_layers):
        ker_reg = K.regularizers.l2(lambtha)
        out = K.layers.Dense(units=layers[i],
                             activation=activations[i],
                             kernel_regularizer=ker_reg)(out)

        if i != n_layers - 1:
            out = K.layers.Dropout(1 - keep_prob)(out)

    model = K.Model(inputs=inputs, outputs=out)

    return model
