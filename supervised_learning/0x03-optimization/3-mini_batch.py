#!/usr/bin/env python3
"""Function train_mini_batch."""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data

'''
def create_batch(X, batch_size):
    """Function to create batches from a data set."""
    m = X.shape[0]
    n_batches = int(m / batch_size)

    batches_list = []
    for i in range(0, n_batches):
        a = i * batch_size
        b = a + batch_size
        X_mini = X[a:b]
        batches_list.append(X_mini)

    if m % batch_size != 0:
        r = m % batch_size

        a = n_batches * batch_size
        b = a + r
        X_mini = X[a:b]
        batches_list.append(X_mini)

    return batches_list
'''


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Function that trains a loaded neural network
    model using mini-batch gradient descent."""
    saved = tf.train.import_meta_graph("{}.meta".format(load_path))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saved.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        for epoch in range(epochs + 1):
            accuracy_t = sess.run(accuracy, feed_dict={x: X_train,
                                                       y: Y_train})
            loss_value_t = sess.run(loss, feed_dict={x: X_train,
                                                     y: Y_train})
            accuracy_v = sess.run(accuracy, feed_dict={x: X_valid,
                                                       y: Y_valid})
            loss_value_v = sess.run(loss, feed_dict={x: X_valid,
                                                     y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_value_t))
            print("\tTraining Accuracy: {}".format(accuracy_t))
            print("\tValidation Cost: {}".format(loss_value_v))
            print("\tValidation Accuracy: {}".format(accuracy_v))
            if epoch < epochs:
                X, Y = shuffle_data(X_train, Y_train)
                # batches_x = create_batch(X, batch_size)
                # batches_y = create_batch(Y, batch_size)
                m = X_train.shape[0]
                step = 1
                for i in range(0, m, batch_size):
                    b_x = X[i: i + batch_size]
                    b_y = Y[i: i + batch_size]
                    sess.run(train_op, feed_dict={x: b_x,
                                                  y: b_y})
                    if step % 100 == 0:
                        accuracy_t, loss_value_t = sess.run((accuracy, loss),
                                                            feed_dict={x: b_x,
                                                                       y: b_y})
                        print("\tStep {}".format(step))
                        print("\t\tCost: {}".format(loss_value_t))
                        print("\t\tAccuracy: {}".format(accuracy_t))

                    step += 1

                if i + batch_size != m:
                    b_x = X[i + batch_size: m]
                    b_y = Y[i + batch_size: m]
                    sess.run(train_op, feed_dict={x: b_x,
                                                  y: b_y})
                    if step % 100 == 0:
                        accuracy_t, loss_value_t = sess.run((accuracy, loss),
                                                            feed_dict={x: b_x,
                                                                       y: b_y})
                        print("\tStep {}".format(step))
                        print("\t\tCost: {}".format(loss_value_t))
                        print("\t\tAccuracy: {}".format(accuracy_t))

        save_path = saver.save(sess, save_path)
        return save_path
