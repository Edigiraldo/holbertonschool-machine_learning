#!/usr/bin/env python3
"""Function l2_reg_create_layer."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer
    that includes L2 regularization."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
         kernel_initializer=init,
         kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha))(prev)

    return layer
