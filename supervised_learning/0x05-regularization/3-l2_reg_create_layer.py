#!/usr/bin/env python3
"""Function l2_reg_create_layer."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer
    that includes L2 regularization."""
    kern_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kern_regul = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=kern_init,
                            kernel_regularizer=kern_regul)(prev)

    return layer
