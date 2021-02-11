#!/usr/bin/env python3
"""Function resnet50."""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015)."""
    he_normal = K.initializers.he_normal()

    x = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=he_normal)(x)
    conv1 = K.layers.BatchNormalization(axis=-1)(conv1)
    conv1 = K.layers.Activation('relu')(conv1)
    conv1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))(conv1)

    conv2_1 = projection_block(conv1, [64, 64, 256], 1)
    conv2_2 = identity_block(conv2_1, [64, 64, 256])
    conv2_3 = identity_block(conv2_2, [64, 64, 256])

    conv3_1 = projection_block(conv2_3, [128, 128, 512])
    conv3_2 = identity_block(conv3_1, [128, 128, 512])
    conv3_3 = identity_block(conv3_2, [128, 128, 512])
    conv3_4 = identity_block(conv3_3, [128, 128, 512])

    conv4_1 = projection_block(conv3_4, [256, 256, 1024])
    conv4_2 = identity_block(conv4_1, [256, 256, 1024])
    conv4_3 = identity_block(conv4_2, [256, 256, 1024])
    conv4_4 = identity_block(conv4_3, [256, 256, 1024])
    conv4_5 = identity_block(conv4_4, [256, 256, 1024])
    conv4_6 = identity_block(conv4_5, [256, 256, 1024])

    conv5_1 = projection_block(conv4_6, [512, 512, 2048])
    conv5_2 = identity_block(conv5_1, [512, 512, 2048])
    conv5_3 = identity_block(conv5_2, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(conv5_3)

    out = K.layers.Dense(units=1000,
                         activation='softmax',
                         kernel_initializer=he_normal)(avg_pool)
    model = K.Model(inputs=x, outputs=out)

    return model
