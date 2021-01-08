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
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            accuracy_t, loss_value_t = sess.run((accuracy, loss), feed_dict={x: X_train, y: Y_train})
            accuracy_v, loss_value_v = sess.run((accuracy, loss), feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_value_t))
                print("\tTraining Accuracy: {}".format(accuracy_t))
                print("\tValidation Cost: {}".format(loss_value_v))
                print("\tValidation Accuracy: {}".format(accuracy_v))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        saver.save(sess, save_path)

    return save_path
