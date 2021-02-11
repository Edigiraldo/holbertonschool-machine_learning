#!/usr/bin/env python3
"""Function projection_block."""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Function that builds a projection block as described
       in Deep Residual Learning for Image Recognition (2015).

       - A_prev is the output from the previous layer
       - filters is a tuple or list containing
         F11, F3, F12, respectively:
         - F11 is the number of filters in the first 1x1 convolution
         - F3 is the number of filters in the 3x3 convolution
         - F12 is the number of filters in the second 1x1
           convolution as well as the 1x1 convolution
           in the shortcut connection
         - s is the stride of the first convolution in both
           the main path and the shortcut connection
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal()

    # Starts path for shorcut connection.
    scl = K.layers.Conv2D(filters=F12,
                          strides=(s, s),
                          kernel_size=(1, 1),
                          padding='valid',
                          kernel_initializer=he_normal)(A_prev)
    sclBN = K.layers.BatchNormalization(axis=-1)(scl)

    # Starts main path.
    l1x1 = K.layers.Conv2D(filters=F11,
                           strides=(s, s),
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

    # Joining paths.
    shorcut = K.layers.Add()([BN1x1_2, sclBN])
    Relu1x1_2 = K.layers.Activation('relu')(shorcut)

    return Relu1x1_2
