import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce
import importlib
from tqdm import tqdm
from scipy.signal import correlate

cifar10 = importlib.import_module("tensorflow.keras.datasets.cifar10")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_normalized = x_train.astype(np.float32) / 255.0

I, J, C = 3, 3, 3

K = [np.random.randn(I, J, C) * np.sqrt(2. / 27) for _ in range(16)]
K = np.stack(K).astype(np.float32)


W1 = np.random.randn(16 * 16 * 16, 128) * np.sqrt(2. / 4096)
b1 = np.zeros((1, 128))

W2  = np.random.randn(128, 10) * np.sqrt(2. / 128)
b2 = np.zeros((1, 10))

def filter_forward_scipy(K, x):
    x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.stack([
        correlate(x_pad, K[k][None], mode='valid', method='direct')[..., 0]
        for k in range(16)    
    ], axis=1)
    return out

def ReLu(z):
    a = np.maximum(0, z)
    return a

def pool(AF):
    N = AF.shape[0]
    blocks = AF.reshape(N, 16, 16, 2, 16, 2)
    pooled = blocks.max(axis=(3, 5))
    return pooled

def upsample(a):
    return np.repeat(np.repeat(a, 2, axis=-2), 2, axis=-1)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shifted)
    exp = exp / np.sum(exp, axis=1, keepdims=True)
    return exp

def y_onehot(y, num_classes = 10):
    m = y.shape[0]
    onehot = np.zeros((m, num_classes))
    onehot[np.arange(m), y.astype(int)] = 1
    return onehot
    
def cross_entry_loss(onehot, a):
    m = onehot.shape[0]
    epsilon = 1e-9
    loss = -np.sum(onehot * np.log(a + epsilon)) / m
    return loss

def forwardpass(x_batch, K):
    #filter stage
    N = x_batch.shape[0]
    ZF = filter_forward_scipy(K, x_batch)
    AF = ReLu(ZF)
    AFP = pool(AF)
    mask = (AF == upsample(AFP))
    #dAF = upsample(dA_pooled) * mask
    # A_p1 -> MLP
    Z1 = AFP.reshape(N, -1) @ W1 + b1
    A1 = ReLu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return A2, Z2, A1, Z1, mask, AFP, AF, ZF

epochs = 10
BATCH = 64
LR = 0.05

for epoch in range(epochs):
    perm = np.random.permutation(len(x_normalized))

    for start in range(0, len(perm), BATCH):
        idx = perm[start:start + BATCH]
        x = x_normalized[idx]
        y = y_train[idx].flatten()
    
        A2, Z2, A1, Z1, mask, AFP, AF, ZF = forwardpass(x, K)
        onehot = y_onehot(y)

        loss = cross_entry_loss(onehot, A2)
        print("epoch: ",epoch, " | ", "loss: ", loss)

        N = x.shape[0]
        x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0))) 

        delta2 = (A2 - onehot) / N
        dW2 = A1. T @ delta2
        db2 = delta2.sum(axis=0, keepdims=True)

        delta1 = (delta2 @ W2.T) * (Z1 > 0)
        dW1 = AFP.reshape(N, -1).T @ delta1
        db1 = delta1.sum(axis=0, keepdims=True)

        dAFP = (delta1 @ W1.T).reshape(N, 16, 16, 16)
        dAF = upsample(dAFP) * mask
        dZF = dAF * (ZF > 0)

        dK = np.stack([
            correlate(x_pad, dZF[:, k, :, :, None], mode='valid', method='direct')[0]
            for k in range(16)
        ]).astype(np.float32)    

        K  -= LR * dK
        W1 -= LR * dW1
        b1 -= LR * db1
        W2 -= LR * dW2
        b2 -= LR * db2

np.savez_compressed('mein_netz_weights.npz', w1=W1, bias1=b1, w2=W2, bias2=b2, filter=K)
