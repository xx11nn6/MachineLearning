import numpy as np
import torch
import matplotlib.pyplot as plt

a = np.load(r'./save/red-cnn-denoised-piglet.npy')
print(a.dtype)
print(a.shape)
print(a[1][200:210, 200:210])
# plt.figure()
# plt.imshow(a[2], cmap='gray')
# plt.show()
