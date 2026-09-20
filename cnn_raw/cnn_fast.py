import numpy as np
import scipy.signal
from scipy import ndimage
from skimage.measure import block_reduce

bild_1 = np.random.randn(4, 4)
kernel1 = np.random.randn(3, 3) * np.sqrt(2. / 3)

W1 = np.random.randn(4, 6) * np.sqrt(2. / 6)
b1 = np.zeros((1, 6))

W2  = np.random.rand(6, 2) * np.sqrt(2. / 2)
b2 = np.zeros((1, 2))

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

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shifted)
    exp = exp / np.sum(exp, axis=1, keepdims=True)
    return exp

def forwardpass(X_train, kernel):
    #filter stage
    ZF = filter_forward_scipy(kernel, X_train)
    AF = ReLu(ZF)
    AFP = pool(AF).ravel()
    # A_p1 -> MLP
    print(AFP.shape)
    print(W1.shape)
    Z1 = AFP @ W1 + b1
    A1 = ReLu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return A2, Z2, Z1, A1, AFP, AF, ZF


print(forwardpass(bild_1, kernel1))

# 1 Durchlauf filter forward pass fertig
