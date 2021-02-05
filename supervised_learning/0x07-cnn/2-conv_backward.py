#!/usr/bin/env python3
"""Function conv_backward."""


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over
    a convolutional layer of a neural network."""
    m_z, h_new, w_new, c_newz = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = W.shape
    c_prev, c_new = W.shape

    sh, sw = stride
    ph, pw = 0, 0

    if padding == 'same':
        ph = int(np.ceil(max((h_prev - 1) * sh + kh - h_prev, 0) / 2))
        pw = int(np.ceil(max((w_prev - 1) * sw + kw - w_prev, 0) / 2))
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    dW = np.zeros(W.shape)
    dA_prev = np.zeros(A_prev.shape)
    db = np.zeros(b.shape)
    db[:, :, 0, :] = np.sum(dZ, axis=(0, 1, 2))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for cn in range(c_new):
                    aux_W = W[:, :, :, cn]
                    aux_dZ = dZ[i, h, w, cn]
                    dA_prev[i,
                            h * sh:kh + (h * sh),
                            w * sw:kw + (w * sw)] += (aux_W * aux_dZ)
                    dW[:, :, :, cn] += (A_prev[i, h * sh:kh + (h * sh),
                                        w * sw:kw + (w * sw)] * aux_dZ)

    _, h_dA, w_dA, _ = dA_prev.shape
    dA_prev = dA_prev[:, ph:h_dA-ph, pw:w_dA-pw, :]

    return dA_prev, dW, db
