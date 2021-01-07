#!/usr/bin/env python3
"""Function create_layer."""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Function to create a layer."""
    Heetal = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=Heetal, activation=activation, name='layer')
    y_pred = layer(prev)

    return y_pred
