#!/usr/bin env python3
"""Class RNNDecoder."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Class to decode for machine translation."""
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor. Sets the following public instance attributes:
            - embedding - a keras Embedding layer that converts
              words from the vocabulary into an embedding vector.
            - gru - a keras GRU layer with units units.
                - Should return both the full sequence of
                  outputs as well as the last hidden state.
                - Recurrent weights should be initialized
                  with glorot_uniform.
            - F - a Dense layer with vocab units.

        - vocab is an integer representing the size of the output
          vocabulary.
        - embedding is an integer representing the dimensionality
          of the embedding vector.
        - units is an integer representing the number of hidden
          units in the RNN cell.
        - batch is an integer representing the batch size.
        """
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def __call__(self, x, s_prev, hidden_states):
        """
        - x is a tensor of shape (batch, 1) containing the previous
          word in the target sequence as an index of the target
          vocabulary.
        - s_prev is a tensor of shape (batch, units) containing the
          previous decoder hidden state.
        - hidden_states is a tensor of shape (batch, input_seq_len,
          units)containing the outputs of the encoder.

        Returns: y, s.
            - y is a tensor of shape (batch, vocab) containing the
              output word as a one hot vector in the target
              vocabulary.
            - s is a tensor of shape (batch, units) containing the
              new decoder hidden state.
        """
        self_attention = SelfAttention(self.units)
        context, weights = self_attention(s_prev, hidden_states)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], -1)
        output, s = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        y = self.F(output)

        return y, s
