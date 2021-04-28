#!/usr/bin/env python3
"""Class LSTMCell."""
import numpy as np


class LSTMCell:
    """Class that represents an LSTM unit."""
    def __init__(self, i, h, o):
        """
        Class constructor.

        - i is the dimensionality of the data.
        - h is the dimensionality of the hidden state.
        - o is the dimensionality of the outputs.

        Creates the public instance attributes Wf, Wu, Wc, Wo,
        Wy, bf, bu, bc, bo, by that represent the weights and
        biases of the cell.
            - Wf and bf are for the forget gate.
            - Wu and bu are for the update gate.
            - Wc and bc are for the intermediate cell state.
            - Wo and bo are for the output gate.
            - Wy and by are for the outputs.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Function that performs forward propagation for one time
        step.

        - x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell.
            - m is the batche size for the data.
        - h_prev is a numpy.ndarray of shape (m, h) containing
          the previous hidden state.
        - c_prev is a numpy.ndarray of shape (m, h) containing
          the previous cell state.

        Returns: h_next, c_next, y.
            - h_next is the next hidden state.
            - c_next is the next cell state.
            - y is the output of the cell.
        """
        hx_prev = np.concatenate((h_prev, x_t), axis=-1)  # (m, i + h)
        Wf, Wu, Wc, Wo, Wy = self.Wf, self.Wu, self.Wc, self.Wo, self.Wy
        bf, bu, bc, bo, by = self.bf, self.bu, self.bc, self.bo, self.by

        ft = self.sigmoid(hx_prev @ Wf + bf)  # (m, h)
        ut = self.sigmoid(hx_prev @ Wu + bu)  # (m, h)
        ot = self.sigmoid(hx_prev @ Wo + bo)  # (m, h)

        Cct = np.tanh(hx_prev @ Wc + bc)   # (m, h)

        Ct = ft * c_prev + ut * Cct
        ht = ot * np.tanh(Ct)
        y = self.softmax(ht @ Wy + by)

        return ht, Ct, y

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
