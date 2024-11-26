import numpy as np
import matplotlib.pyplot as plt
import os


def display_datasets(output_paths):
    print("Loading and displaying datasets...")
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))  # 2 行 10 列
    flag = False
    for row, output_path in enumerate(output_paths):
        # 加载数据集
        datasets = np.load(output_path)
        print(f"Loaded {output_path}, shape: {datasets.shape}")
        if flag == False:
            selected_indices = np.random.choice(
                len(datasets), 10, replace=False)
            flag = True
        # 显示前 10 张图像
        for col in range(10):
            ax = axes[row, col]
            ax.imshow(datasets[selected_indices[col]], cmap='gray')  # 灰度显示
            ax.axis('off')  # 不显示坐标轴

        # 在每一行的开头添加标记
        if row == 0:
            axes[row, 0].set_title("Full Dose CT")
        else:
            axes[row, 0].set_title("Quarter Dose CT")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    current_dir = os.getcwd()
    # 输出目录
    out_dir = r'dataset\piglet_npy'
    output_names = ['full_dose_ct.npy', 'quarter_dose_ct.npy']
    output_paths = []

    # 遍历每个剂量目录
    for output_name in output_names:
        output_path = os.path.join(out_dir, output_name)
        output_paths.append(output_path)

    # 显示两个数据集的前 10 张图像
    display_datasets(output_paths)
