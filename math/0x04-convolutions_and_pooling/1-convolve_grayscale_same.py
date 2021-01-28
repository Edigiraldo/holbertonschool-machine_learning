#!/usr/bin/env python3
"""Function convolve_grayscale_same."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same
    convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_along_height = max((h - 1) + kh - h, 0)
    pad_along_width = max((w - 1) + kw - w, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    images_padded = np.zeros((m,
                             h + pad_along_height,
                             w + pad_along_width))
    images_padded[:, pad_top:-pad_bottom, pad_left:-pad_right] = images

    out_h = h
    out_w = w

    out = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            img = images_padded[:, i: i + kh, j: j + kw]
            out[:, i, j] = (kernel[np.newaxis, :] * img)\
                .sum(axis=2).sum(axis=1)

    return out
