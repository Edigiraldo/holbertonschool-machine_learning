#!/usr/bin/env python3
"""Function autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder.

    - input_dims is an integer containing the dimensions of
      the model input.
    - hidden_layers is a list containing the number of nodes
      for each hidden layer in the encoder, respectively.
    - latent_dims is an integer containing the dimensions
      of the latent space representation.

    Returns: encoder, decoder, auto
        - encoder is the encoder model, which should output the
          latent representation, the mean, and the log variance,
          respectively.
        - decoder is the decoder model.
        - auto is the full autoencoder model.
    """
    backend = keras.backend

    def sampling(args):
        z_mean, z_log_sigma = args
        batch = backend.shape(z_mean)[0]
        epsilon = backend.random_normal(shape=(batch, latent_dims),
                                        mean=0.0, stddev=0.1)
        return z_mean + backend.exp(z_log_sigma) * epsilon

    encoder_In = keras.Input(shape=(input_dims,))
    encoder = encoder_In
    for nodes in hidden_layers:
        encoder = keras.layers.Dense(nodes, activation='relu')(encoder)

    z_mean = keras.layers.Dense(latent_dims)(encoder)
    z_log_sigma = keras.layers.Dense(latent_dims)(encoder)

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    decoder_In = keras.Input(shape=(latent_dims,))
    decoder = decoder_In
    for nodes in hidden_layers[::-1]:
        decoder = keras.layers.Dense(nodes, activation='relu')(decoder)
    decoder = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)

    encoder = keras.Model(encoder_In, [z, z_mean, z_log_sigma])
    decoder = keras.Model(decoder_In, decoder)

    out = decoder(encoder(encoder_In))
    auto = keras.Model(encoder_In, out)

    def vae_loss(val1, val2):
        reconstruction_loss = keras.losses.binary_crossentropy(encoder_In, out)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma
        kl_loss = kl_loss - backend.square(z_mean) - backend.exp(z_log_sigma)
        kl_loss = backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = backend.mean(reconstruction_loss + kl_loss)
        return vae_loss
    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
