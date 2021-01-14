#!/usr/bin/env python3
"""Function learning_rate_decay."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that creates a learning rate decay operation
    in tensorflow using inverse time decay."""
    decay_lr = tf.train.inverse_time_decay(learning_rate=alpha,
                                           global_step=global_step,
                                           decay_steps=decay_step,
                                           decay_rate=decay_rate,
                                           staircase=True)

    return decay_lr
