#!/usr/bin/env python3
"""Functions save_weights and load_weights."""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Function that saves the model’s weights."""
    network.save_weights(filename)


def load_weights(network, filename):
    """Function that loads the model’s weights"""
    network.load_weights(filename)
