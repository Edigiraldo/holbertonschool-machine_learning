#!/usr/bin/env python3
"""Function from_numpy."""
import pandas as pd


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray.

    - array is the np.ndarray from which you should create the pd.DataFrame.

    Returns: the newly created pd.DataFrame.
    """
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n_cols = array.shape[1]
    cols = list(alphabet[:n_cols])
    return pd.DataFrame(array, columns=cols)
