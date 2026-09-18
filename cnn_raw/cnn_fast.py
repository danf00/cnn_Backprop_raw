import numpy as np
import scipy.signal
from scipy import ndimage
from skimage.measure import block_reduce
bild_1 = np.random.randn(4, 4)
kernel1 = np.random.randn(3, 3) * np.sqrt(2. / 3)


def filter_forward_scipy(kernel, bild):
    output = scipy.signal.correlate2d(bild, kernel, mode='same')
    return output

filter_bild = filter_forward_scipy(kernel1, bild_1)

def ReLu(z):
    a = np.maximum(0, z)
    return a

relu_bild = ReLu(filter_bild)

def pool(bild):
    out_max = block_reduce(bild, block_size=(2, 2), func=np.max)
    return out_max

pool_bild = pool(relu_bild)

print(bild_1)

print(filter_bild)

print(relu_bild)

print(pool_bild)

# 1 Durchlauf filter forward pass fertig
