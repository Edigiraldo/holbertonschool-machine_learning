#!/usr/bin/env python3
"""Function create_RMSProp_op"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Function that creates the training operation
    for a neural network in tensorflow using the
    RMSProp optimization algorithm."""
    optimizer = tf.train.RMSPropOptimizer(alpha, momentum=beta2, epsilon=epsilon)
    train = optimizer.minimize(loss)

    return train
