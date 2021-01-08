#!/usr/bin/env python3
"""Forward propagation function."""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward
    propagation graph for the neural network."""
    prev = x
    for i in range(len(layer_sizes)):
        n = layer_sizes[i]
        activation = activations[i]
        prev = create_layer(prev, n, activation)

    return prev
