#!/usr/bin/env python3
"""Function autoencoder."""
import tensorflow.keras as keras
# from tensorflow import keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Function that creates an sparse autoencoder.

    - input_dims is an integer containing the dimensions of the
      model input.
    - hidden_layers is a list containing the number of nodes for
      each hidden layer in the encoder, respectively.
        - the hidden layers should be reversed for the decoder.
    - latent_dims is an integer containing the dimensions of the
      latent space representation.
    - lambtha is the regularization parameter used for L1
      regularization on the encoded output.

    Returns: encoder, decoder, auto.
        - encoder is the encoder model.
        - decoder is the decoder model.
        - auto is the sparse autoencoder model.
    """
    L1 = keras.regularizers.l1(lambtha)

    encoder_In = keras.Input(shape=(input_dims,))
    encoder = encoder_In
    for nodes in hidden_layers:
        encoder = keras.layers.Dense(nodes, activation='relu',
                                     activity_regularizer=L1)(encoder)
    encoder = keras.layers.Dense(latent_dims, activation='relu',
                                 activity_regularizer=L1)(encoder)

    decoder_In = keras.Input(shape=(latent_dims,))
    decoder = decoder_In
    for nodes in hidden_layers[::-1]:
        decoder = keras.layers.Dense(nodes, activation='relu',
                                     activity_regularizer=L1)(decoder)
    decoder = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)

    encoder = keras.Model(encoder_In, encoder)
    decoder = keras.Model(decoder_In, decoder)
    auto = keras.Model(encoder_In, decoder(encoder(encoder_In)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
