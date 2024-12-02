import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

b = np.load(r'./save/mayo_cyclegan_denoised.npy')
a = np.load(r'./save/mayo_full_dose.npy')
c = np.load(r'./save/mayo_low_dose.npy')
# index = 325
# p1 = psnr(a[index], b[index])
# p2 = psnr(b[index], c[index])
# print(p1, p2)
# print(a.dtype)
# print(a.shape)
# print(np.min(a))
# print(np.max(a))
plt.figure()
plt.subplot(121)
plt.imshow(a[10], cmap='gray')
plt.subplot(122)
plt.imshow(c[10], cmap='gray')
plt.show()

# a = (a-np.min(a))/(np.max(a)-np.min(a))
# np.save(os.path.join(r'./save', 'mayo_redcnn_denoised.npy'), a.astype(np.float16))
# b = np.load(r'./save/mayo_red-cnn_denoised.npy')
# c = np.load(r'./dataset/mayo_npy/L067_2_input.npy')

# input_files = sorted([f for f in os.listdir(
#     r'./dataset/mayo_npy') if ('L506' in f) and ('_input.npy' in f)])
# num_images = len(input_files)
# imgs = np.zeros((num_images, 512, 512), dtype=np.float32)
# for i in range(num_images):
#     imgs[i] = np.load(os.path.join(
#         r'./dataset/mayo_npy', input_files[i])).astype(np.float32)

# print(a.dtype)
# print(len(c))
# print(np.min(a))
# print(np.max(a))
# plt.figure()
# i = 8
# plt.subplot(151)
# plt.imshow(a[i], cmap='gray')
# plt.subplot(152)
# plt.imshow(a[i+1], cmap='gray')
# plt.subplot(153)
# plt.imshow(a[i+2], cmap='gray')
# plt.subplot(154)
# plt.imshow(a[i+3], cmap='gray')
# plt.subplot(155)
# plt.imshow(a[i+4], cmap='gray')
# plt.subplot(122)
# plt.imshow(c, cmap='gray')
plt.show()
