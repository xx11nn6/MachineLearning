import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import numpy as np
import os


def display_datasets(input_paths):
    show_size = 5
    print("Loading and displaying datasets...")
    fig, axes = plt.subplots(
        2, show_size, figsize=(show_size*2, 4))  # 2 行 10 列
    flag = False
    for row, input_path in enumerate(input_paths):
        # 加载数据集
        datasets = np.load(input_path)
        print(f"Loaded {input_path}, shape: {datasets.shape}")
        if flag == False:
            selected_indices = np.random.choice(
                len(datasets), show_size, replace=False)
            flag = True
        # 显示前 10 张图像
        for col in range(show_size):
            img = datasets[selected_indices[col]]
            ax = axes[row, col]

            ax.imshow(img, cmap='gray')  # 灰度显示
            ax.axis('off')  # 不显示坐标轴

        # 在每一行的开头添加标记
        if row == 0:
            axes[row, 0].set_title("Full Dose CT")
        else:
            axes[row, 0].set_title("Quarter Dose CT")

    plt.tight_layout()
    plt.show()


def zoom_with_psnr_ssim(input_paths, output_dir):
    size = 2
    selected_indices = np.random.choice(850, size, replace=False)
    full_dose_datasets = np.load(input_paths[0])
    quarter_dose_datasets = np.load(input_paths[1])

    # 获取图像
    full_dose_imgs = full_dose_datasets[selected_indices]
    quarter_dose_imgs = quarter_dose_datasets[selected_indices]

    group = np.stack(
        [np.stack(full_dose_imgs), np.stack(quarter_dose_imgs)], axis=1)

    # 待修改参数
    color = (1, 0, 0)
    # 定义局部放大区域的坐标
    x1, y1 = 180, 180
    x2, y2 = 220, 220
    linewidth = 2
    zoom_array = [0.55, 0.10, 0.3, 0.3]   # x y 子图左下角相对于整个画幅的位置， 后两位子图大小

    # print(group.shape)
    # 遍历每个图像
    for i in range(size):
        full_dose_img = group[i, 0]
        quarter_dose_img = group[i, 1]
        psnr = PSNR(full_dose_img, quarter_dose_img)
        ssim = SSIM(full_dose_img, quarter_dose_img)

        for j in range(2):
            img = group[i, j]
            # 获取局部放大区域的图像数据
            zoomed_region = img[y1:y2, x1:x2]
            fig, ax = plt.subplots()

            # 绘制原始图像
            ax.imshow(img, cmap='gray')
            plt.text(0, 0.01, f'PSNR={psnr:.3f}', transform=plt.gca().transAxes,
                     color='white', fontsize=12)
            plt.text(0, 0.06, f'SSIM={ssim:.3f}', transform=plt.gca().transAxes,
                     color='white', fontsize=12)
            ax.axis('off')

            # 绘制局部放大区域的边界框
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=linewidth, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # 创建放大图
            zoom_ax = fig.add_axes(zoom_array)
            zoom_ax.imshow(zoomed_region, cmap='gray')
            # 设置放大图的边框，并增加线宽
            zoom_ax.spines['top'].set_color(color)
            zoom_ax.spines['top'].set_linewidth(linewidth)
            zoom_ax.spines['bottom'].set_color(color)
            zoom_ax.spines['bottom'].set_linewidth(linewidth)
            zoom_ax.spines['left'].set_color(color)
            zoom_ax.spines['left'].set_linewidth(linewidth)
            zoom_ax.spines['right'].set_color(color)
            zoom_ax.spines['right'].set_linewidth(linewidth)
            # 隐藏放大图的坐标轴
            zoom_ax.set_xticks([])
            zoom_ax.set_yticks([])

            # 绘制线连接原始图像和放大图
            line = ConnectionPatch(xyA=(x1, y2), xyB=(0, 0), coordsA="data", coordsB="axes fraction",
                                   axesA=ax, axesB=zoom_ax, color=color, linewidth=linewidth-1)
            fig.add_artist(line)
            line2 = ConnectionPatch(xyA=(x2, y1), xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                                    axesA=ax, axesB=zoom_ax, color=color, linewidth=linewidth-1)
            fig.add_artist(line2)

            # plt.show()
            # 保存
            if j == 0:
                save_path = os.path.join(
                    out_dir, f"zoomed_full_{selected_indices[i]}")
            else:
                save_path = os.path.join(
                    out_dir, f"zoomed_quarter_{selected_indices[i]}")
            plt.savefig(save_path, transparent=True,
                        dpi=300, bbox_inches='tight')


def PSNR(img1, img2):
    m = img1.shape[0]
    n = img2.shape[1]
    diff = img1-img2
    MSE = 1/m*1/n*np.sum(diff**2)
    PSNR = 10*np.log10(1/MSE)
    # print(PSNR)
    return PSNR


def SSIM(img1, img2):
    u1 = np.mean(img1)
    u2 = np.mean(img2)
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    c1 = np.square(0.01*7)
    c2 = np.square(0.05*7)
    SSIM = (2*u1*u2+c1)*(sigma1*sigma2+c2)/((u1**2+u2**2+c1)*(var1+var2+c2))
    # print(SSIM)
    return SSIM


if __name__ == '__main__':
    current_dir = os.getcwd()
    # 数据集目录
    in_dir = r'dataset\piglet_npy'
    input_names = ['full_dose_ct.npy', 'quarter_dose_ct.npy']
    input_paths = []

    # 输出目录
    out_dir = r'images\raw_ssim_psnr'

    # 遍历每个剂量目录
    for input_name in input_names:
        input_path = os.path.join(in_dir, input_name)
        input_paths.append(input_path)

    # 显示两个数据集的前 10 张图像
    # display_datasets(input_paths)
    zoom_with_psnr_ssim(input_paths, out_dir)
