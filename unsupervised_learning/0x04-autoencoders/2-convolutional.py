#!/usr/bin/env python3
"""Function autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Function that creates a convolutional autoencoder.

    - input_dims is a tuple of integers containing the dimensions
      of the model input.
    - filters is a list containing the number of filters for each
      convolutional layer in the encoder, respectively.
    - latent_dims is a tuple of integers containing the dimensions
      of the latent space representation.

    Returns: encoder, decoder, auto
        - encoder is the encoder model.
        - decoder is the decoder model.
        - auto is the full autoencoder model.
    """
    layers = keras.layers
    encoder_In = keras.Input(shape=input_dims)

    encoder = encoder_In
    for num_f in filters:
        encoder = layers.Conv2D(num_f, (3, 3), activation='relu',
                                padding='same')(encoder)
        encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    decoder_In = keras.Input(shape=latent_dims)

    decoder = decoder_In
    for num_f in filters[:0:-1]:
        decoder = layers.Conv2D(num_f, (3, 3), activation='relu',
                                padding='same')(decoder)
        decoder = layers.UpSampling2D((2, 2))(decoder)

    # last two layers.
    decoder = layers.Conv2D(filters[0], (3, 3), activation='relu',
                            padding='valid')(decoder)
    decoder = layers.UpSampling2D((2, 2))(decoder)
    decoder = layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                            padding='same')(decoder)

    encoder = keras.Model(encoder_In, encoder)
    decoder = keras.Model(decoder_In, decoder)
    auto = keras.Model(encoder_In, decoder(encoder(encoder_In)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
