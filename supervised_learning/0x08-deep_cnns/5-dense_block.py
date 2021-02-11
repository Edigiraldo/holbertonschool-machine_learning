#!/usr/bin/env python3
"""Function dense_block."""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block as described
       in Densely Connected Convolutional Networks.

       - X is the output from the previous layer.
       - nb_filters is an integer representing the number
         of filters in X.
       - growth_rate is the growth rate for the dense block.
       - layers is the number of layers in the dense block.
    """
    x = X
    for i in range(layers):
        conv_out = H(x, growth_rate * 4, 1)
        out = H(conv_out, growth_rate, 3)
        x = K.layers.Concatenate(axis=-1)([x, out])

        nb_filters += growth_rate

    return x, nb_filters


def H(inputs, n_filters, kern_size=3):
    """Function representing layer."""
    he_normal = K.initializers.he_normal()

    x = K.layers.BatchNormalization(axis=-1)(inputs)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters=n_filters,
                        kernel_size=(kern_size, kern_size),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=he_normal)(x)

    return x
