#!/usr/bin/env python3
"""Function convolve_grayscale."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Function that performs a convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if type(padding) == tuple:
        ph, pw = padding

    elif padding == "same":
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    elif padding == "valid":
        ph, pw = 0, 0

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    out_h = int(((h + (2*ph) - kh) / sh) + 1)
    out_w = int(((w + (2*pw) - kw) / sw) + 1)
    out = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            img = images_padded[:, i * sh: i * sh + kh, j * sw: j * sw + kw]
            out[:, i, j] = (kernel[np.newaxis, :] * img)\
                .sum(axis=2).sum(axis=1)

    return out
