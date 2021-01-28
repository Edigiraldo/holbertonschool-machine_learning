#!/usr/bin/env python3
"""Function convolve_grayscale_same."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same
    convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = max((kh - 1) // 2,
             kh // 2)
    pw = max((kw - 1) // 2,
             kw // 2)

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    out_h = h
    out_w = w

    out = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            img = images_padded[:, i: i + kh, j: j + kw]
            out[:, i, j] = (kernel[np.newaxis, :] * img)\
                .sum(axis=2).sum(axis=1)

    return out
