#!/usr/bin/env python3
"""evaluate function."""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Funciton that evaluates the
    output of a neural network."""
    saved = tf.train.import_meta_graph("{}.meta".format(save_path))
    with tf.Session() as sess:
        saved.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        y_pred, accuracy, loss = sess.run(
            [y_pred, accuracy, loss],
            feed_dict={x: X, y: Y})

        return y_pred, accuracy, loss
