#!/usr/bin/env python3
"""Function convolve_grayscale_valid."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid
    convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    out = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            img = images[:, i: i + kh, j: j + kw]
            out[:, i, j] = (kernel[np.newaxis, :] * img)\
                .sum(axis=2).sum(axis=1)

    return out
