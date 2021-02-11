#!/usr/bin/env python3
"""Function transition_layer."""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a transition layer as described
       in Densely Connected Convolutional Networks.

       - X is the output from the previous layer.
       - nb_filters is an integer representing
         the number of filters in X.
       - compression is the compression factor
         for the transition layer.
    """
    he_normal = K.initializers.he_normal()

    x = K.layers.BatchNormalization(axis=-1)(X)
    x = K.layers.Activation('relu')(x)

    conv_filters = int(nb_filters * compression)

    x = K.layers.Conv2D(filters=conv_filters,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=he_normal)(x)
    x = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  padding='same')(x)

    return x, conv_filters
