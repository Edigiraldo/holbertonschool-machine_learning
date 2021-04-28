#!/usr/bin/env python3
"""Function deep_rnn."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Function that performs forward propagation for a deep RNN.

    - rnn_cells is a list of RNNCell instances of length l that
      will be used for the forward propagation.
        - l is the number of layers.
    - X is the data to be used, given as a numpy.ndarray of
      shape (t, m, i).
        - t is the maximum number of time steps.
        - m is the batch size.
        - i is the dimensionality of the data.
    - h_0 is the initial hidden state, given as a numpy.ndarray
      of shape (l, m, h).
        - h is the dimensionality of the hidden state.

    Returns: H, Y.
        - H is a numpy.ndarray containing all of the hidden states
          of shape (t, l, m, h).
        - Y is a numpy.ndarray containing all of the outputs of
          shape (t, m, o).
    """
    l, _, h = h_0.shape
    t_max, m, i = X.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((1, l, m, h))
    H[0, :, :, :] = h_0

    Y = np.zeros((t_max, m, o))

    for t in range(t_max):
        H_t = np.zeros((l, m, h))
        for lay in range(l):
            rnn_cell = rnn_cells[lay]

            h_prev = H[t, lay, :, :]  # (m, h)

            if lay == 0:
                x_t = X[t, :, :]  # (m , i)

            h_next, y = rnn_cell.forward(h_prev, x_t)  # (m, h), (m, o)

            # x_t for next layer same time.
            x_t = h_next  # (m, h)

            H_t[lay, :, :] = h_next

        H = np.append(H, H_t[np.newaxis, :, :, :], axis=0)
        Y[t, :, :] = y

    return H, Y
