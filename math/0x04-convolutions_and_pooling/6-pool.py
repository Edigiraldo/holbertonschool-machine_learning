#!/usr/bin/env python3
"""Function pool."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = int(((h - kh) / sh) + 1)
    out_w = int(((w - kw) / sw) + 1)
    out = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            img = images[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
            if mode == "max":
                out[:, i, j, :] = img.max(axis=(1, 2))
            elif mode == "avg":
                out[:, i, j, :] = img.mean(axis=(1, 2))

    return out
