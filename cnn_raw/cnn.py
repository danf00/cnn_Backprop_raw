import numpy as np

##this is a test commit

#initailisierung von den startgewichten eines conv1 filters

bild = np.random.randn(4, 4)
bild = np.pad(bild, pad_width=1, mode='constant', constant_values=0)
print("bild alt: ")
print(bild)
neu_bild = []
conv1 = np.random.randn(3, 3) * np.sqrt(2. / 3)
print("filter: ")
print(conv1)
neu_bild = np.zeros((4, 4))
for z in range(4):
    for y in range(4):
        neu_pixel_wert = 0
        for i in range(3):
            for j in range(3):
                neu_pixel_wert += bild[z+1-1+i][y+1-1+j]*conv1[i][j]
        neu_bild[z][y] = neu_pixel_wert
print("bild nach filter")
print(neu_bild)
