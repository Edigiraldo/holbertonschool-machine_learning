#!/usr/bin/env python3
"""train function."""
import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            _, accuracy_t, loss_value_t = sess.run((train_op, accuracy, loss), feed_dict={x: X_train, y: Y_train})
            accuracy_v, loss_value_v = sess.run((accuracy, loss), feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0:
                print("After {i} iterations:")
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_value_t))
                print("\tTraining Accuracy: {}".format(accuracy_t))
                print("\tValidation Cost: {}".format(loss_value_v))
                print("\tValidation Accuracy: {}".format(accuracy_v))

        saver.save(sess, save_path)

    return save_path
