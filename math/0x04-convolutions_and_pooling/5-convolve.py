#!/usr/bin/env python3
"""Function convolve."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Function that performs a convolution
    on images using multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if type(padding) == tuple:
        ph, pw = padding

    elif padding == "same":
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    elif padding == "valid":
        ph, pw = 0, 0

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           'constant')

    out_h = int(((h + (2*ph) - kh) / sh) + 1)
    out_w = int(((w + (2*pw) - kw) / sw) + 1)
    out = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            img = images_padded[:, i * sh: i * sh + kh,
                                j * sw: j * sw + kw, :, np.newaxis]
            out[:, i, j] = (kernels[np.newaxis, :] * img)\
                .sum(axis=1).sum(axis=2).sum(axis=1)

    return out