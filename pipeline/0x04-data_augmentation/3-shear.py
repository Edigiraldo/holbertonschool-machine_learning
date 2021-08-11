#!/usr/bin/env python3
"""Function shear."""
import tensorflow as tf


def shear_image(image, intensity):
    """
    Function that randomly shears an image.

    - Image is a 3D tf.Tensor containing the image to shear.
    - Intensity is the intensity with which the image should be sheared.

    Returns the sheared image.
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(img, intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)
    sheared_img = tf.keras.preprocessing.image.array_to_img(sheared)

    return sheared_img
