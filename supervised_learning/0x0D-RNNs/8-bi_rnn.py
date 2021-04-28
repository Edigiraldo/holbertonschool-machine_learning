#!/usr/bin/env python3
"""Function bi_rnn."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Function that performs forward propagation for a
    bidirectional RNN.

    - bi_cell is an instance of BidirectionalCell that will be
      used for the forward propagation.
    - X is the data to be used, given as a numpy.ndarray of
      shape (t, m, i).
        - t is the maximum number of time steps.
        - m is the batch size.
        - i is the dimensionality of the data.
    - h_0 is the initial hidden state in the forward direction,
      given as a numpy.ndarray of shape (m, h).
        - h is the dimensionality of the hidden state.
    - h_t is the initial hidden state in the backward direction,
      given as a numpy.ndarray of shape (m, h).

    Returns: H, Y.
        H is a numpy.ndarray containing all of the concatenated
        hidden states.
        Y is a numpy.ndarray containing all of the outputs.
    """
    t_max, m, i = X.shape
    _, h = h_0.shape

    Hf = np.zeros((1, m, h))
    Hf[0, :, :] = h_0
    for t in range(t_max):
        h_prev = Hf[t]
        x_t = X[t]
        h_next = bi_cell.forward(h_prev, x_t)

        Hf = np.append(Hf, h_next[np.newaxis, :, :], axis=0)

    Hb = np.zeros((1, m, h))
    Hb[0, :, :] = h_t
    for t in range(t_max - 1, -1, -1):
        h_next = Hb[0]
        x_t = X[t]
        h_pev = bi_cell.backward(h_next, x_t)

        Hb = np.append(h_pev[np.newaxis, :, :], Hb, axis=0)

    Hf, Hb = Hf[1:], Hb[0:-1]
    H = np.concatenate((Hf, Hb), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
