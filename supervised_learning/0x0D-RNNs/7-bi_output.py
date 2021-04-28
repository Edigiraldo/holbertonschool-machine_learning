#!/usr/bin/env python3
"""Class BidirectionalCell."""
import numpy as np


class BidirectionalCell:
    """Class that represents a bidirectional cell of an RNN."""
    def __init__(self, i, h, o):
        """
        Class constructor.

        - i is the dimensionality of the data.
        - h is the dimensionality of the hidden states.
        - o is the dimensionality of the outputs.

        Creates the public instance attributes Whf, Whb, Wy, bhf,
        bhb, by that represent the weights and biases of the cell.
            - Whf and bhfare for the hidden states in the forward
              direction.
            - Whb and bhbare for the hidden states in the backward
              direction.
            - Wy and byare for the outputs.
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Function that calculates the hidden state in the
        forward direction for one time step.

        - x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell.
            - m is the batch size for the data.
        - h_prev is a numpy.ndarray of shape (m, h) containing
          the previous hidden state.

        Returns: h_next, the next hidden state.
        """
        hx_t = np.concatenate((h_prev, x_t), axis=-1)
        Whf, bhf = self.Whf, self.bhf
        h_next = np.tanh(hx_t @ Whf + bhf)  # (m, h)

        return h_next

    def backward(self, h_next, x_t):
        """
        Function that calculates the hidden state in the backward.
        direction for one time step.

        - x_t is a numpy.ndarray of shape (m, i) that contains the
          data input for the cell.
            - m is the batch size for the data.
        - h_next is a numpy.ndarray of shape (m, h) containing the
          next hidden state.

        Returns: h_pev, the previous hidden state.
        """
        hx_t = np.concatenate((h_next, x_t), axis=-1)
        Whb, bhb = self.Whb, self.bhb
        h_pev = np.tanh(hx_t @ Whb + bhb)  # (m, h)

        return h_pev

    def output(self, H):
        """
        Method that calculates all outputs for the RNN.

        - H is a numpy.ndarray of shape (t, m, 2 * h) that
          contains the concatenated hidden states from both
          directions, excluding their initialized states.
            - t is the number of time steps.
            - m is the batch size for the data.
            - h is the dimensionality of the hidden states.

        Returns: Y, the outputs.
        """
        Wy, by = self.Wy, self.by  # (2h, o), (1, o)

        Y = np.matmul(H, Wy) + by[np.newaxis, :, :]  # (t, m, o)

        for i in range(len(Y)):
            Y[i] = self.softmax(Y[i])

        return Y

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x ~ (m, o)."""
        maxx = np.max(x, axis=1).reshape(-1, 1)
        e_x = np.exp(x - maxx)

        summ = e_x.sum(axis=1).reshape(-1, 1)

        return e_x / summ
