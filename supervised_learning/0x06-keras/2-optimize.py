#!/usr/bin/env python3
"""Function optimize_model."""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Function that sets up Adam optimization
    for a keras model with categorical crossentropy
    loss and accuracy metrics."""
    adam = K.optimizers.Adam(lr=alpha,
                             beta_1=beta1,
                             beta_2=beta2)

    cce = 'categorical_crossentropy'

    network.compile(optimizer=adam,
                    loss=cce,
                    metrics=['accuracy'])
