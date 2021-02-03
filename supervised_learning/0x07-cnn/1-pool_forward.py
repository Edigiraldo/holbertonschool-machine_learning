#!/usr/bin/env python3
"""Function pool_forward."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propagation
    over a pooling layer of a neural network."""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = int(((h - kh) / sh) + 1)
    out_w = int(((w - kw) / sw) + 1)
    out = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            img = A_prev[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
            if mode == "max":
                out[:, i, j, :] = img.max(axis=(1, 2))
            elif mode == "avg":
                out[:, i, j, :] = img.mean(axis=(1, 2))

    return out
