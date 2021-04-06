import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def conv2d_fast(img, krn):
    is0, is1, ks0, ks1 = *img.shape, *krn.shape
    rs0, rs1 = is0 - ks0 + 1, is1 - ks1 + 1
    
    ix0 = np.arange(ks0)[:, None] + np.arange(rs0)[None, :]
    ix1 = np.arange(ks1)[:, None] + np.arange(rs1)[None, :]
    
    res = krn[:, None, :, None] * img[(ix0.ravel()[:, None], 
                ix1.ravel()[None, :])].reshape(ks0, rs0, ks1, rs1)
    res = res.transpose(1, 3, 0, 2).reshape(rs0, rs1, -1).sum(axis = -1)
    
    return res

def conv_batch(I, proto_I):
    Hi, Wi = I[:,:,0].shape
    Hk, Wk = proto_I[:,:,0].shape
    hk = Hk//2
    wk = Wk//2

    # padding
    new_img = np.pad(I[:,:,0], (hk, wk), 'constant', constant_values=0)
    pHi, pWi = new_img.shape

    out = np.zeros((Hi, Wi))
    for i in range(hk, pHi-hk):
        for j in range(wk, pWi-wk):
            batch = new_img[i-hk:i+hk+1, j-wk:j+wk+1]
            out[i-hk][j-wk] = np.sum(batch*proto_I[:,:,0])