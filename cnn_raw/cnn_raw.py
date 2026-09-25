import numpy as np
import math
import importlib
import random
import time
from tqdm import tqdm
cifar10 = importlib.import_module("tensorflow.keras.datasets.cifar10")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_normalized = x_train.astype(np.float32) / 255.0
#np.set_printoptions(suppress=True, precision=6)

bild_1 = np.random.randn(4, 4)
bild_nullen = np.pad(bild_1, pad_width=1, mode='constant', constant_values=0)

label = np.array([2])

#conv layer festlegen 
kernel1 = np.random.randn(3, 3) * np.sqrt(2. / 27)
kernel2 = np.random.randn(3, 3) * np.sqrt(2. / 27)
kernel3 = np.random.randn(3, 3) * np.sqrt(2. / 27)
kernel4 = np.random.randn(3, 3) * np.sqrt(2. / 27)

I, J  = kernel1.shape

filter  = [kernel1, kernel2, kernel3, kernel4]
NUM_FILTER = len(filter)

W1 = np.random.randn(1024, 64) * np.sqrt(2./ 1024)
b1 = np.zeros((1, 64))

W2 = np.random.randn(64, 10) * np.sqrt(2. / 64)
b2 = np.zeros((1, 10))

def filter_forward_schleifen(kernel, bild):
    output = []
    for t in kernel:
        neu_bild = np.zeros((X, Y))
        for z in range(X):
            for y in range(Y):
                neu_pixel_wert = 0
                for i in range(I):
                    for j in range(J):
                        for p in range(3):
                            neu_pixel_wert += bild[z+1-1+i][y+1-1+j][p]*t[i][j]
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

def relu_derivative(z):
    return (z > 0).astype(float)

def y_onehot(y, num_classes = 10):
    m = y.shape[0]
    onehot = np.zeros((m, num_classes))
    onehot[np.arange(m), y.astype(int)] = 1
    return onehot

def cross_entry_loss(onehot, A3):
    m = onehot.shape[0]
    epsilon = 1e-9
    loss = -np.sum(onehot * np.log(A3 + epsilon)) / m
    return loss

    
def forwardpass(X_train, kernel):
    ZF = filter_forward_schleifen(kernel, X_train)
    AF = ReLuConv(ZF)
    A_pics, A_cords = pooling(AF)
    A_pics = np.array(A_pics).flatten().reshape(1, -1)
    #A_p1 -> MLP
    Z1 =  A_pics @ W1 + b1
    A1 = ReLuMLP(Z1)
    Z2 = A1 @ W2 + b2 
    A2 = softmax(Z2)

    return A2, Z2, Z1, A1, A_pics, A_cords, AF, ZF

#Backpropagation

BATCH_SIZE = 32
learning_rate = 0.01
epochs = 10000 // BATCH_SIZE

pbar = tqdm(range(epochs), desc="Training", unit="epoch")

for epoch in pbar:

    dW1_batch = np.zeros_like(W1)
    db1_batch = np.zeros_like(b1)
    dW2_batch = np.zeros_like(W2)
    db2_batch = np.zeros_like(b2)
    dKernels_batch = [np.zeros((I, J)) for _ in range(NUM_FILTER)]
    loss_batch = 0.0

    batch_indices = np.random.randint(0, 49999, size=BATCH_SIZE)
    batch_indices = np.array([123, 123, 123])
    #rndm = 987
    for rndm in batch_indices:
        pic = x_normalized[rndm]
        X, Y = pic.shape[0], pic.shape[1]
        pic_nullen = np.pad(
            pic, 
            pad_width=((1, 1), (1, 1), (0, 0)),
            mode='constant',
            constant_values=0
        )

        onehot = y_onehot(y_train[rndm])

        A2, Z2, Z1, A1, A_pics, A_cords, AF, ZF = forwardpass(pic_nullen, filter)
        loss = cross_entry_loss(onehot, A2)
        loss_batch += loss
        #print("Epoch: ", epoch, " | Loss: ", loss)

        delta2 = A2 - onehot
        dW2 = A1.T @ delta2
        db2 = delta2

        delta1 = (delta2 @ W2.T) * relu_derivative(Z1)
        dW1 = A_pics.T @ delta1
        db1 = delta1

        dA_pics = delta1 @ W1.T
        dA_pics = dA_pics.reshape(4, 16, 16)
        dAF = [np.zeros((X, Y)) for _ in range(len(filter))]

        for k in range(4):
            for i in range(X//2):
                for j in range(Y//2):
                    z, y = A_cords[k][i][j]
                    dAF[k][int(z)][int(y)] = dA_pics[k][i][j]

        dZF = [dAF[k] * relu_derivative(ZF[k]) for k in range(4)]

        dKernels = []
        for k in range(4):
            dK  = np.zeros((I, J))
            for i in range(I):
                for j in range(J):
                    wert = 0
                    for z in range(X):
                        for y in range(Y):
                            for p in range(3):
                                wert += pic_nullen[z+i][y+j][p] * dZF[k][z][y]
                    dK[i][j] = wert
            dKernels.append(dK)

        dW1_batch += dW1
        db1_batch += db1
        dW2_batch += dW2
        db2_batch += db2

    dW1_batch /= BATCH_SIZE
    db1_batch /= BATCH_SIZE
    dW2_batch /= BATCH_SIZE
    db2_batch /= BATCH_SIZE
    for k in range(NUM_FILTER):
        dKernels_batch[k] /= BATCH_SIZE
    loss_batch /= BATCH_SIZE


    if epoch % 10 == 0:
        pbar.set_postfix({"loss": loss_batch})

    for k in range(4):
        filter[k] = filter[k] - learning_rate * dKernels[k]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

print("------Training fertig------")
print("Model loss | ", loss)