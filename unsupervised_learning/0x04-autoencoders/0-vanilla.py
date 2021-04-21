#!/usr/bin/env python3
"""Function autoencoder."""
from tensorflow import keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates an autoencoder.

    - input_dims is an integer containing the dimensions of the
      model input.
    - hidden_layers is a list containing the number of nodes for
      each hidden layer in the encoder, respectively.
    - the hidden layers should be reversed for the decoder.
    - latent_dims is an integer containing the dimensions of the
      latent space representation.

    Returns: encoder, decoder, auto.
        - encoder is the encoder model.
        - decoder is the decoder model.
        - auto is the full autoencoder model.
    """
    encoder_In = keras.Input(shape=(input_dims,))
    encoder = encoder_In
    for nodes in hidden_layers:
        encoder = keras.layers.Dense(nodes, activation='relu')(encoder)
    encoder = keras.layers.Dense(latent_dims, activation='relu')(encoder)

    decoder_In = keras.Input(shape=(latent_dims,))
    decoder = decoder_In
    for nodes in hidden_layers[::-1]:
        decoder = keras.layers.Dense(nodes, activation='relu')(decoder)
    decoder = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)

    encoder = keras.Model(encoder_In, encoder)
    decoder = keras.Model(decoder_In, decoder)
    auto = keras.Model(encoder_In, decoder(encoder(encoder_In)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
