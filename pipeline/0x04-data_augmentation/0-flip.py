#!/usr/bin/env python3
"""Function flip_image."""
import tensorflow as tf


def flip_image(image):
    """
    Function that flips an image horizontally.

    - Image is a 3D tf.Tensor containing the image to flip.

    Returns the flipped image.
    """
    fliped_img = tf.image.flip_left_right(image)

    return fliped_img
