import numpy as np
import math

bild_1 = np.random.randn(4, 4)
bild_nullen = np.pad(bild_1, pad_width=1, mode='constant', constant_values=0)
X, Y = bild_1.shape

kernel1 = np.random.randn(3, 3) * np.sqrt(2. / 3)
I, J  = kernel1.shape

def filter_forward_schleifen(kernel, bild):
    neu_bild = np.zeros((X, Y))
    for z in range(X):
        for y in range(Y):
            neu_pixel_wert = 0
            for i in range(I):
                for j in range(J):
                    neu_pixel_wert += bild[z+1-1+i][y+1-1+j]*kernel[i][j]
            neu_bild[z][y] = neu_pixel_wert
    return neu_bild

filter_bild = filter_forward_schleifen(kernel1, bild_nullen)

def ReLu(filter_bild):
    test_bild = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            if filter_bild[x][y] < 0:
                test_bild[x][y] = 0
            else:
                test_bild[x][y] = filter_bild[x][y]
    return test_bild

relu_bild = ReLu(filter_bild)

def pooling(bild):
    pooled = np.zeros((int(X/2), int(Y/2)))
    for i in range(int(X/2)):
        for j in range(int(Y/2)):
            wert = -math.inf
            if bild[i*2][j*2] > wert:
                wert = bild[i*2][j*2]
            if bild[i*2][j*2+1] > wert:
                wert = bild[i*2][j*2+1]
            if bild[i*2+1][j*2] > wert:
                wert = bild[i*2+1][j*2]
            if bild[i*2+1][j*2+1] > wert:
                wert = bild[i*2+1][j*2+1]
            pooled[i][j] = wert
    return pooled
pooled_bild = pooling(relu_bild)


print(filter_bild)
print(relu_bild)
print(pooled_bild)

        
