#!/usr/bin/env python3
"""calculate_loss function."""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Function that calculates the softmax
    cross-entropy loss of a prediction."""
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
