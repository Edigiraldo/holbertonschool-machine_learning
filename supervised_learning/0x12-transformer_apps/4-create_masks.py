#!/usr/bin/env python3
"""Function create_masks."""
import tensorflow as tf


def create_look_ahead_mask(size):
    """
    Creates look ahead mask.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    """
    Creates padding mask.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks(inputs, target):
    """
    Function that creates all masks for training/validation.

    - inputs is a tf.Tensor of shape (batch_size, seq_len_in) that contains
      the input sentence.
    - target is a tf.Tensor of shape (batch_size, seq_len_out) that contains
      the target sentence.

    Returns: encoder_mask, combined_mask, decoder_mask.
        - encoder_mask is the tf.Tensor padding mask of shape
          (batch_size, 1, 1, seq_len_in) to be applied in the encoder.
        - combined_mask is the tf.Tensor of shape
          (batch_size, 1, seq_len_out, seq_len_out) used in the 1st attention
          block in the decoder to pad and mask future tokens in the input
          received by the decoder. It takes the maximum between a lookaheadmask
          and the decoder target padding mask.
        - decoder_mask is the tf.Tensor padding mask of shape
          (batch_size, 1, 1, seq_len_in) used in the 2nd attention block in the
          decoder.
    """
    enc_padding_mask = create_padding_mask(inputs)
    dec_padding_mask = create_padding_mask(inputs)

    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
