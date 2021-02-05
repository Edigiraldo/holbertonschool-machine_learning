#!/usr/bin/env python3
"""Function pool_backward."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs back propagation
    over a pooling layer of a neural network."""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dx = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for cn in range(c_new):
                    if mode == 'max':
                        aux = A_prev[i,
                                     h*sh:kh+(h*sh),
                                     w*sw:kw+(w*sw),
                                     cn]
                        mask = (aux == np.max(aux))
                        dx[i,
                           h*sh:kh+(h*sh),
                           w*sw:kw+(w*sw),
                           cn] += dA[i, h, w, cn] * mask
                    if mode == 'avg':
                        dx[i,
                           h*sh:kh+(h*sh),
                           w*sw:kw+(w*sw),
                           cn] += (dA[i, h, w, cn])/kh/kw

    return dx
