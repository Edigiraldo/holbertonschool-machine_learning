#!/usr/bin/env python3
"""Function """
import tensorflow as tf


def lenet5(x, y):
    """Function that builds a modified version
    of the LeNet-5 architecture using tensorflow."""
    Henormal = tf.contrib.layers.variance_scaling_initializer()

    C1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                          activation=tf.nn.relu, padding="same",
                          kernel_initializer=Henormal)(x)
    MP1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=(2, 2))(C1)

    C2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                          activation=tf.nn.relu, padding="valid",
                          kernel_initializer=Henormal)(MP1)
    MP2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=(2, 2))(C2)

    Flat = tf.layers.Flatten()(MP2)

    D1 = tf.layers.Dense(units=120, kernel_initializer=Henormal,
                         activation=tf.nn.relu)(Flat)
    D2 = tf.layers.Dense(units=84, kernel_initializer=Henormal,
                         activation=tf.nn.relu)(D2)
    logits = tf.layers.Dense(units=10, kernel_initializer=Henormal)(D1)

    y_pred = tf.nn.softmax(logits)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=y_pred)

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    y_true = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))

    return y_pred, train, loss, acc
