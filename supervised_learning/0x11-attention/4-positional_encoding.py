#!/usr/bin/env python3
"""Function positional_encoding."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a
    transformer.

    - max_seq_len is an integer representing the maximum
      sequence length.
    - dm is the model depth.

    Returns: a numpy.ndarray of shape (max_seq_len, dm)
    containing the positional encoding vectors.
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    d = np.arange(dm)[np.newaxis, :]

    PE = pos / np.power(10000, (2 * (d//2)) / np.float32(dm))

    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])

    return PE
