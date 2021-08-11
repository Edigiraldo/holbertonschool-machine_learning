#!/usr/bin/env python3
"""Function crop_image."""
import tensorflow as tf


def crop_image(image, size):
    """
    Function that performs a random crop of an image.

    - image is a 3D tf.Tensor containing the image to crop.
    - size is a tuple containing the size of the crop.

    Returns the cropped image.
    """
    croped = tf.random_crop(image, size)

    return croped
