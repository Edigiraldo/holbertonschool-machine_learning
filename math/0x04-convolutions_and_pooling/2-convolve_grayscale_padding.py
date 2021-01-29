#!/usr/bin/env python3
"""Function ."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Function that performs a convolution
    on grayscale images with custom padding."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    out = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            img = images_padded[:, i: i + kh, j: j + kw]
            out[:, i, j] = (kernel[np.newaxis, :] * img)\
                .sum(axis=2).sum(axis=1)

    return out
