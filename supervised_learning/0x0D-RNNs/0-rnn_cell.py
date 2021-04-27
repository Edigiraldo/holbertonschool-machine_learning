#!/usr/bin/env python3
"""Class RNNCell."""
import numpy as np


class RNNCell:
    """Class that represents a cell of a simple RNN."""
    def __init__(self, i, h, o):
        """
        Class constructor.

        - i is the dimensionality of the data.10
        - h is the dimensionality of the hidden state.15
        - o is the dimensionality of the outputs.5
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Function that performs forward propagation for one time
        step.

        - x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell.
            - m is the batche size for the data.
        - h_prev is a numpy.ndarray of shape (m, h) containing
          the previous hidden state.

        Returns: h_next, y.
            - h_next is the next hidden state.
            - y is the output of the cell.
        """
        hx_t = np.concatenate((h_prev, x_t), axis=-1)
        Wh, Wy, bh, by = self.Wh, self.Wy, self.bh, self.by

        h_next = np.tanh(hx_t @ Wh + bh)  # (m, h)
        y = self.softmax(h_next @ Wy + by)  # (m, o)

        return h_next, y

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x ~ (m, o)."""
        maxx = np.max(x, axis=1).reshape(-1, 1)
        e_x = np.exp(x - maxx)

        summ = e_x.sum(axis=1).reshape(-1, 1)

        return e_x / summ
