#!/usr/bin/env python3
"""Functions save_config and load_config."""
import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s
    configuration in JSON format."""
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


def load_config(filename):
    """Function that loads a model
    with a specific configuration."""
    with open(filename, "r") as f:
        config = f.read()
        loaded_model = K.models.model_from_json(config)
    return loaded_model
