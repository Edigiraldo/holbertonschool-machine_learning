#!/usr/bin/env python3
"""Function train_mini_batch."""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_batch(X, batch_size):
    """Function to create batches from a data set."""

    batch_list = []
    i = 0
    m = X.shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    for b in range(batches):
        if b != batches - 1:
            batch_list.append(X[i:(i + batch_size)])
        else:
            batch_list.append(X[i:])
        i += batch_size

    return batch_list


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Function that trains a loaded neural network
    model using mini-batch gradient descent."""
    saved = tf.train.import_meta_graph("{}.meta".format(load_path))
    with tf.Session() as sess:
        saved.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        for i in range(epochs + 1):
            accuracy_t, loss_value_t = sess.run((accuracy, loss),
                                                feed_dict={x: X_train,
                                                           y: Y_train})
            accuracy_v, loss_value_v = sess.run((accuracy, loss),
                                                feed_dict={x: X_valid,
                                                           y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(loss_value_t))
            print("\tTraining Accuracy: {}".format(accuracy_t))
            print("\tValidation Cost: {}".format(loss_value_v))
            print("\tValidation Accuracy: {}".format(accuracy_v))
            if i < epochs:
                X, Y = shuffle_data(X_train, Y_train)
                batches_x = create_batch(X, batch_size)
                batches_y = create_batch(Y, batch_size)
                for i, b_x in enumerate(batches_x):
                    b_y = batches_y[i]
                    sess.run(train_op, feed_dict={x: b_x,
                                                  y: b_y})
                    if (i + 1) % 100 == 0 and i != 0:
                        accuracy_t, loss_value_t = sess.run((accuracy, loss),
                                                            feed_dict={x: b_x,
                                                                       y: b_y})
                        print("\tStep {}".format(i + 1))
                        print("\t\tCost: {}".format(loss_value_t))
                        print("\t\tAccuracy: {}".format(accuracy_t))

        save_path = saved.save(sess, save_path)
        return save_path
