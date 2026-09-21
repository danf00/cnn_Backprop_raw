import numpy as np
import math

np.set_printoptions(suppress=True, precision=6)

bild_1 = np.random.randn(4, 4)
bild_nullen = np.pad(bild_1, pad_width=1, mode='constant', constant_values=0)
X, Y = bild_1.shape

label = np.array([2])

#conv layer festlegen 
kernel1 = np.random.randn(3, 3) * np.sqrt(2. / 3)
kernel2 = np.random.randn(3, 3) * np.sqrt(2. / 3)
kernel3 = np.random.randn(3, 3) * np.sqrt(2. / 3)
kernel4 = np.random.randn(3, 3) * np.sqrt(2. / 3)

I, J  = kernel1.shape

filter  = [kernel1, kernel2, kernel3, kernel4]

W1 = np.random.randn(16, 32) * np.sqrt(2./ 32)
b1 = np.zeros((1, 32))

W2 = np.random.randn(32, 4) * np.sqrt(2. / 4)
b2 = np.zeros((1, 4))

def filter_forward_schleifen(kernel, bild):
    output = []
    for t in kernel:
        neu_bild = np.zeros((X, Y))
        for z in range(X):
            for y in range(Y):
                neu_pixel_wert = 0
                for i in range(I):
                    for j in range(J):
                        neu_pixel_wert += bild[z+1-1+i][y+1-1+j]*t[i][j]
                neu_bild[z][y] = neu_pixel_wert
        output.append(neu_bild)
    return output

def ReLuConv(filter_bild):
    output = []
    for l in filter_bild:
        test_bild = np.zeros((X, Y))
        for x in range(X):
            for y in range(Y):
                if l[x][y] < 0:
                    test_bild[x][y] = 0
                else:
                    test_bild[x][y] = l[x][y]
        output.append(test_bild)
    return output

def ReLuMLP(z):
    a = np.maximum(0, z)
    return a


def pooling(bild):
    pooled_pics_list = []
    pooled_cords_list = []
    for p in bild:
        pooled = np.zeros((int(X/2), int(Y/2)))
        pooled_cords = np.zeros((int(X/2), int(Y/2), 2))
        for i in range(int(X/2)):
            for j in range(int(Y/2)):
                wert = -math.inf
                if p[i*2][j*2] > wert:
                    wert = p[i*2][j*2]
                    pooled_cords[i][j] = [i*2, j*2]
                if p[i*2][j*2+1] > wert:
                    wert = p[i*2][j*2+1]
                    pooled_cords[i][j] = [i*2, j*2+1]
                if p[i*2+1][j*2] > wert:
                    wert = p[i*2+1][j*2]
                    pooled_cords[i][j] = [i*2+1, j*2]
                if p[i*2+1][j*2+1] > wert:
                    wert = p[i*2+1][j*2+1]
                    pooled_cords[i][j] = [i*2+1, j*2+1]
                pooled[i][j] = wert
        pooled_pics_list.append(pooled)
        pooled_cords_list.append(pooled_cords)
    return pooled_pics_list, pooled_cords_list

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shifted)
    exp = exp/np.sum(exp, axis=1, keepdims=True)
    return exp

def y_onehot(y, num_classes = 4):
    print(y)
    m = y.shape[0]
    onehot = np.zeros((m, num_classes))
    onehot[np.arange(m), y.astype(int)] = 1
    return onehot

def cross_entry_loss(onehot, A3):
    m = onehot.shape[0]
    epsilon = 1e-12
    loss = -np.sum(onehot * np.log(A3 + epsilon)) / m
    return loss

    
def forwardpass(X_train, kernel):
    ZF = filter_forward_schleifen(kernel, X_train)
    AF = ReLuConv(ZF)
    A_pics, A_cords = pooling(AF)
    #A_p1 -> MLP
    Z1 =  np.array(A_pics).ravel() @ W1 + b1
    A1 = ReLuMLP(Z1)
    Z2 = A1 @ W2 + b2 
    A2 = softmax(Z2)

    return A2, Z2, Z1, A1, A_pics, A_cords, AF, ZF

A2, Z2, Z1, A1, A_pics, A_cords, AF, ZF = forwardpass(bild_nullen, filter)

print(A2)
onehot = y_onehot(label)
loss = cross_entry_loss(onehot, A2)
print(loss)

