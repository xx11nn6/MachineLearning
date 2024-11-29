import pydicom
import numpy as np
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt


def normalize_dicom(dicom_raw):
    # dicom CT数据归一化
    window_center = dicom_raw[0x0028, 0x1050].value  # 窗位
    window_width = dicom_raw[0x0028, 0x1051].value  # 窗宽
    intercept = dicom_raw[0x0028, 0x1052].value  # 截距
    slope = dicom_raw[0x0028, 0x1053].value  # 斜率
    img = dicom_raw.pixel_array.copy()

    lower_value = window_center - window_width / 2
    norm_img = img * slope + intercept
    norm_img = (norm_img - lower_value) / window_width
    norm_img = np.clip(norm_img, 0, 1)
    return norm_img


def pre_proccess(root_in, root_out):
    # 预处理，输入数据集路径、保存路径
    datasets = []
    filelist = glob.glob("{}/*".format(root_in))
    print('%d files in the folder, start preprocessing' % len(filelist))

    # 提取文件的切片厚度信息
    slices = []
    for filename in filelist:
        dicom_raw = pydicom.dcmread(filename)
        z_position = dicom_raw.ImagePositionPatient[2]  # 切片厚度
        slices.append((filename, z_position))

    # 按切片厚度从小到大排序
    sorted_slices = sorted(slices, key=lambda x: x[1], reverse=True)
    # 选取第101到第860张图像(剔除全黑或数据少的图像)
    selected_slices = sorted_slices[100:860]

    # 逐个处理排序后的文件
    for filename, _ in tqdm(selected_slices):
        dicom_raw = pydicom.dcmread(filename)
        # DICOM归一化
        norm_img = normalize_dicom(dicom_raw)
        datasets.append(norm_img)
    datasets = np.array(datasets, dtype=np.float16)
    print('Pre-processing finished, saving to', root_out)

    # 保存为.npy 文件
    np.save(root_out, datasets)
    print(f"Saved datasets to {root_out}, shape: {datasets.shape}")


def display_datasets(output_paths):
    # 随机显示10张图像
    print("Loading and displaying datasets...")
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))  # 3 行 5 列
    flag = False
    for row, output_path in enumerate(output_paths):
        # 加载数据集
        datasets = np.load(output_path)
        print(f"Loaded {output_path}, shape: {datasets.shape}")
        if flag == False:
            selected_indices = np.random.choice(
                len(datasets), 5, replace=False)
            flag = True
        # 显示前 5 张图像
        for col in range(5):
            ax = axes[row, col]
            ax.imshow(datasets[selected_indices[col]], cmap='gray')  # 灰度显示
            ax.axis('off')  # 不显示坐标轴

        # 在每一行的开头添加标记
        if row == 0:
            axes[row, 0].set_title("Full Dose CT")
        elif row == 1:
            axes[row, 0].set_title("Quarter Dose CT")
        else:
            axes[row, 0].set_title("One Tenth Dose CT")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 输入目录
    input_dirs = [
        r'dataset\piglet\DICOM\PA0\ST0\SE4',  # FULL DOSE FBP
        r'dataset\piglet\DICOM\PA0\ST0\SE14',  # 25 DOSE FBP
        r'dataset\piglet\DICOM\PA0\ST0\SE19'  # 10 DOSE FBP
    ]

    # 输出目录
    out_dir = r'dataset\piglet_npy'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  # 如果输出目录不存在则创建
    output_names = ['full_dose_ct.npy',
                    'quarter_dose_ct.npy', 'onetenth_dose_ct.npy']

    output_paths = []

    # 遍历每个剂量目录
    for input_dir, output_name in zip(input_dirs, output_names):
        input_path = os.path.join(current_dir, input_dir)
        output_path = os.path.join(out_dir, output_name)
        output_paths.append(output_path)

        pre_proccess(input_path, output_path)

    # 显示两个数据集的前 10 张图像
    display_datasets(output_paths)
