#!/usr/bin/env python3
"""Function lenet5."""
import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of
    the LeNet-5 architecture using keras."""
    he_ini = K.initializers.he_normal()
    con_l1 = K.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation='relu',
                             kernel_initializer=he_ini)(X)
    l1_pool = K.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2))(con_l1)
    l2_con = K.layers.Conv2D(filters=16,
                             kernel_size=(5, 5), padding='valid',
                             activation='relu',
                             kernel_initializer=he_ini)(l1_pool)
    l2_pool = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(l2_con)
    flat = K.layers.Flatten()(l2_pool)
    l3_fully = K.layers.Dense(units=120,
                              activation='relu',
                              kernel_initializer=he_ini)(flat)
    l4_fully = K.layers.Dense(units=84,
                              activation='relu',
                              kernel_initializer=he_ini)(l3_fully)
    l5_fully = K.layers.Dense(units=10,
                              activation='softmax',
                              kernel_initializer=he_ini)(l4_fully)
    nn = K.Model(inputs=X, outputs=l5_fully)
    nn.compile(optimizer=K.optimizers.Adam(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

    return nn
