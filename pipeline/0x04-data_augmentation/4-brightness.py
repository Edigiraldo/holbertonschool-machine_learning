#!/usr/bin/env python3
"""Function change_brightness."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Function that randomly changes the brightness of an image.

    - Image is a 3D tf.Tensor containing the image to change.
    - Max_delta is the maximum amount the image should be brightened
      (or darkened).

    Returns the altered image.
    """
    adjusted = tf.image.adjust_brightness(image, max_delta)

    return adjusted
