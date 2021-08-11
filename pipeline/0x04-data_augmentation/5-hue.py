#!/usr/bin/env python3
"""Function change_hue."""
import tensorflow as tf


def change_hue(image, delta):
    """
    Function that changes the hue of an image.
    - image is a 3D tf.Tensor containing the image to change.
    - delta is the amount the hue should change.

    Returns the altered image.
    """
    adjusted = tf.image.adjust_hue(image, delta)

    return adjusted
