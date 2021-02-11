#!/usr/bin/env python3
"""Function identity_block."""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Function that builds an identity block as described in
       Deep Residual Learning for Image Recognition (2015).
       - A_prev is the output from the previous layer.
       - filters is a tuple or list containing.
         F11, F3, F12, respectively:
             - F11 is the number of filters in the first 1x1 convolution.
             - F3 is the number of filters in the 3x3 convolution.
             - F12 is the number of filters in the second 1x1 convolution.
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal()

    l1x1 = K.layers.Conv2D(filters=F11,
                           kernel_size=(1, 1),
                           padding='valid',
                           kernel_initializer=he_normal)(A_prev)
    BN1x1 = K.layers.BatchNormalization(axis=-1)(l1x1)
    Relu1x1 = K.layers.Activation('relu')(BN1x1)

    l3x3 = K.layers.Conv2D(filters=F3,
                           kernel_size=(3, 3),
                           padding='same',
                           kernel_initializer=he_normal)(Relu1x1)
    BN3x3 = K.layers.BatchNormalization(axis=-1)(l3x3)
    Relu3x3 = K.layers.Activation('relu')(BN3x3)

    l1x1_2 = K.layers.Conv2D(filters=F12,
                             kernel_size=(1, 1),
                             padding='valid',
                             kernel_initializer=he_normal)(Relu3x3)
    BN1x1_2 = K.layers.BatchNormalization(axis=-1)(l1x1_2)
    shorcut = K.layers.Add()([BN1x1_2, A_prev])
    Relu1x1_2 = K.layers.Activation('relu')(shorcut)

    return Relu1x1_2
