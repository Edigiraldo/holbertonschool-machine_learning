#!/usr/bin/env python3
"""Function inception_block."""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Function that builds an inception block.
        - A_prev: is the output from the previous layer
        - filters: is a tuple or list containing
          F1, F3R, F3,F5R, F5, FPP, respectively:
            - F1 is the number of filters in the 1x1 convolution
            - F3R is the number of filters in the 1x1
              convolution before the 3x3 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F5R is the number of filters in the 1x1
              convolution before the 5x5 convolution
            - F5 is the number of filters in the 5x5 convolution
            - FPP is the number of filters in the 1x1
              convolution after the max pooling
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    l1x1 = K.layers.Conv2D(filters=F1,
                           kernel_size=(1, 1),
                           padding='same',
                           activation='relu')(A_prev)

    l1x1_3x3 = K.layers.Conv2D(filters=F3R,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu')(A_prev)
    l3x3 = K.layers.Conv2D(filters=F3,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(l1x1_3x3)

    l1x1_5x5 = K.layers.Conv2D(filters=F5R,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu')(A_prev)
    l5x5 = K.layers.Conv2D(filters=F5,
                           kernel_size=(5, 5),
                           padding='same',
                           activation='relu')(l1x1_5x5)

    lMP_1x1 = K.layers.MaxPool2D(pool_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same')(A_prev)
    l1x1MP = K.layers.Conv2D(filters=FPP,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu')(lMP_1x1)

    concat = K.layers.Concatenate(axis=3)([l1x1, l3x3, l5x5, l1x1MP])

    return concat
