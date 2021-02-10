#!/usr/bin/env python3
"""Function inception_block."""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network v1(GoogleNet)."""
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            padding='same',
                            strides=(2, 2),
                            activation='relu')(X)
    max_pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(conv1)

    l1x1_conv2 = K.layers.Conv2D(filters=64,
                                 kernel_size=(1, 1),
                                 padding='same',
                                 activation='relu')(max_pool1)
    conv2 = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            padding='same',
                            strides=(1, 1),
                            activation='relu')(l1x1_conv2)
    max_pool2 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(conv2)

    inception_3a = inception_block(max_pool2,
                                   [64, 96, 128, 16, 32, 32])

    inception_3b = inception_block(inception_3a,
                                   [128, 128, 192, 32, 96, 64])

    max_pool3 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(inception_3b)

    inception_4a = inception_block(max_pool3,
                                   [192, 96, 208, 16, 48, 64])

    inception_4b = inception_block(inception_4a,
                                   [160, 112, 224, 24, 64, 64])

    inception_4c = inception_block(inception_4b,
                                   [128, 128, 256, 24, 64, 64])

    inception_4d = inception_block(inception_4c,
                                   [112, 144, 288, 32, 64, 64])

    inception_4e = inception_block(inception_4d,
                                   [256, 160, 320, 32, 128, 128])

    max_pool4 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2))(inception_4e)

    inception_5a = inception_block(max_pool4,
                                   [256, 160, 320, 32, 128, 128])

    inception_5b = inception_block(inception_5a,
                                   [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(inception_5b)
    dropout = K.layers.Dropout(0.4)(avg_pool)
    x = K.layers.Dense(units=1000,
                       activation='softmax')(dropout)

    model = K.Model(inputs=X, outputs=x)

    return model
