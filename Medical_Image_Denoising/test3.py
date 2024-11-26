import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
dose_dir=r'dataset\piglet_npy\fifty_dose_ct.npy'
dir=os.path.join(current_dir, dose_dir)
# 加载.npy文件
images = np.load(dir)


# 打印数据类型
print("数据类型:", images.dtype)

# 打印数据类型对应的位深度
if images.dtype == 'uint8':
    print("位深度: 8位")
elif images.dtype == 'uint16':
    print("位深度: 16位")
elif images.dtype == 'float32':
    print("位深度: 32位")
elif images.dtype == 'float64':
    print("位深度: 64位")
else:
    print("未知的位深度")

# 显示前10张图片
fig, axes = plt.subplots(1, 10, figsize=(20, 2))  # 创建一个1行10列的子图布局
for i in range(10):
    ax = axes[i]
    ax.imshow(images[i], cmap='gray')  # 使用灰度色图显示图片
    ax.axis('off')  # 不显示坐标轴
plt.show()
