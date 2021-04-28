#!/usr/bin/env python3
"""Class GRUCell."""
import numpy as np


class GRUCell:
    """Class that represents a gated recurrent unit."""
    def __init__(self, i, h, o):
        """
        Class constructor.

        - i is the dimensionality of the data.
        - h is the dimensionality of the hidden state.
        - o is the dimensionality of the outputs.
        - Creates the public instance attributes Wz, Wr, Wh, Wy, bz,
          br, bh, by that represent the weights and biases of the cell.
            - Wzand bz are for the update gate.
            - Wrand br are for the reset gate.
            - Whand bh are for the intermediate hidden state.
            - Wyand by are for the output.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Function that performs forward propagation for one time
        step.

        - x_t is a numpy.ndarray of shape (m, i) that contains the
          data input for the cell.
            - m is the batche size for the data.
        - h_prev is a numpy.ndarray of shape (m, h) containing the
          previous hidden state.

        Returns: h_next, y.
            - h_next is the next hidden state.
            - y is the output of the cell.
        """
        hx_prev = np.concatenate((h_prev, x_t), axis=-1)  # (m, i + h)
        Wz, Wr, Wh, Wy = self.Wz, self.Wr, self.Wh, self.Wy
        bz, br, bh, by = self.bz, self.br, self.bh, self.by

        zt = self.sigmoid(hx_prev @ Wz + bz)  # (m, h)
        rt = self.sigmoid(hx_prev @ Wr + br)  # (m, h)

        rhx_t = np.concatenate((rt * h_prev, x_t), axis=-1)

        hut = np.tanh(rhx_t @ Wh + bh)  # (m, h)
        ht = (1 - zt) * h_prev + zt * hut  # (m, h)

        y = self.softmax(ht @ Wy + by)  # (m, o)

        return ht, y

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x ~ (m, o)."""
        maxx = np.max(x, axis=1).reshape(-1, 1)
        e_x = np.exp(x - maxx)

        summ = e_x.sum(axis=1).reshape(-1, 1)

        return e_x / summ

    @staticmethod
    def sigmoid(x):
        """Sigmoid"""

        return (1 / (1 + np.exp(-x)))
