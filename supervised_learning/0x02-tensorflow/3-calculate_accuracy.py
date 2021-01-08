#!/usr/bin/env python3
"""calculate_accuracy function."""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the
    accuracy of a prediction."""
    equality = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy
