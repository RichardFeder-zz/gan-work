import numpy as np

def split3d(arr3d, nchunks):
    arr3ds = [arr3d,]
    for di in range(3):
        arr3dsi = []
        for ai in range(len(arr3ds)):
            arr3dsi.extend(np.split(arr3ds[ai], nchunks, axis=di))
        arr3ds = arr3dsi

    del arr3d
    
    return arr3ds

def squash(x, a=4.0):
    
    return 2*x / (x + a) - 1

def squash_inv(y, a=4.0):
    
    return -a*(y+1)/(y-1)
