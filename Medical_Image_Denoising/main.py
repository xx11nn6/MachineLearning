import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前工作目录
current_dir = os.getcwd()
# 定义相对路径
file_path = os.path.join(current_dir, r'dataset\cmayo\train_64.npy')

# 读取.npy文件
data = np.load(file_path)
# 显示数据的大小
print("Size of the data:", data.shape)

# 假设第一个维度是低剂量图像，第二个维度是高剂量图像
# 选取前5个样本
fig, axs = plt.subplots(5, 2, figsize=(10, 20))  # 创建一个图形和子图

for i in range(5):
    # 显示低剂量图像
    axs[i, 0].imshow(data[0, i].squeeze(), cmap='gray')
    axs[i, 0].set_title(f'Low Dose Image {i+1}')
    axs[i, 0].axis('off')

    # 显示高剂量图像
    axs[i, 1].imshow(data[1, i].squeeze(), cmap='gray')
    axs[i, 1].set_title(f'High Dose Image {i+1}')
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()
