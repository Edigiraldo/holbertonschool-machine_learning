#!/usr/bin/env python3
"""Function dropout_create_layer."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that creates a layer of
    a neural network using dropout"""
    kern_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kern_regul = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=kern_init,
                            kernel_regularizer=kern_regul)(prev)

    return layer
