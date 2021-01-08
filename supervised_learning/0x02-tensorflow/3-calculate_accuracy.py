#!/usr/bin/env python3
"""calculate_accuracy function."""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the
    accuracy of a prediction."""
    y_true = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))
    return acc
