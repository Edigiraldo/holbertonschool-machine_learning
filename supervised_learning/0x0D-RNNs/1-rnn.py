#!/usr/bin/env python3
"""Function rnn."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Function that performs forward propagation for a simple RNN.

    - rnn_cell is an instance of RNNCell that will be used for
      the forward propagation.
    - X is the data to be used, given as a numpy.ndarray of
      shape (t, m, i).
        - t is the maximum number of time steps.
        - m is the batch size.
        - i is the dimensionality of the data.
    - h_0 is the initial hidden state, given as a numpy.ndarray
      of shape (m, h).
        - h is the dimensionality of the hidden state.

    Returns: H, Y.
        - H is a numpy.ndarray containing all of the hidden
          states.
        - Y is a numpy.ndarray containing all of the outputs.
    """
    t_max, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((1, m, h))
    H[0, :, :] = h_0[:, :]

    Y = np.zeros((1, m, o))

    h_prev = h_0
    for t in range(0, t_max):
        x_t = X[t]
        h_next, y = rnn_cell.forward(h_prev, x_t)

        H = np.append(H, h_next[np.newaxis, :, :], axis=0)
        Y = np.append(Y, y[np.newaxis, :, :], axis=0)

        h_prev = h_next

    Y = Y[1:]

    return H, Y
