import numpy as np
import scipy.signal
from scipy import ndimage
from skimage.measure import block_reduce
bild_1 = np.random.randn(4, 4)
bild_nullen = np.pad(bild_1, pad_width=1, mode='constant', constant_values=0)
kernel1 = np.random.randn(3, 3) * np.sqrt(2. / 3)

def filter_forward_schleifen(kernel, bild):
    #legt filter über bild 
    neu_bild = np.zeros((4, 4))
    for z in range(4):
        for y in range(4):
            neu_pixel_wert = 0
            for i in range(3):
                for j in range(3):
                    neu_pixel_wert += bild[z+1-1+i][y+1-1+j]*kernel[i][j]
            neu_bild[z][y] = neu_pixel_wert
    return neu_bild

def filter_forward_scipy(kernel, bild):
    output = scipy.signal.correlate2d(bild, kernel, mode='same')
    return output

filter_bild = filter_forward_scipy(kernel1, bild_1)

def ReLu(filter_bild):
    test_bild = np.zeros((4, 4))
    for x in range(4):
        for y in range(4):
            if filter_bild[x][y] < 0:
                test_bild[x][y] = 0
            else:
                test_bild[x][y] = filter_bild[x][y]
    return test_bild

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
