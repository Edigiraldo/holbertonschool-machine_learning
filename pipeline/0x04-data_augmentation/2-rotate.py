#!/usr/bin/env python3
"""Function rotate_image."""
import tensorflow as tf


def rotate_image(image):
    """
    Function that rotates an image by 90 degrees counter-clockwise.

    - image is a 3D tf.Tensor containing the image to rotate

    Returns the rotated image.
    """
    rotated = tf.image.rot90(image, k=1)

    return rotated
