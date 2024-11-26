import pydicom
import numpy as np
import os
from tqdm import tqdm

# 归一化
def normalize_dicom(img, window_center, window_width, intercept, slope):
    lower_value = window_center - window_width / 2
    norm_img = img * slope + intercept
    norm_img = (norm_img - lower_value) / window_width
    norm_img = np.clip(norm_img, 0, 1)
    return norm_img.astype(np.float16)

if __name__ == '__main__':
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 读取dicom文件的元数据(dicom tags)
    dose_dirs = [
        r'dataset\piglet\DICOM\PA0\ST0\SE26',
        r'dataset\piglet\DICOM\PA0\ST0\SE31'
    ]
    output_names = ['full_dose_ct.npy', 'fifty_dose_ct.npy']

    # 输出目录
    out_dir = r'dataset\piglet_npy'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  # 如果输出目录不存在，则创建

    # 遍历每个剂量目录
    for dose_dir, output_name in zip(dose_dirs, output_names):
        file_path = os.path.join(current_dir, dose_dir)
        output_path = os.path.join(out_dir, output_name)

        # 获取所有dcm文件并初始化列表来存储所有归一化后的图像
        dcm_files = [f for f in os.listdir(file_path)]
        all_norm_imgs = []

        # 使用tqdm显示进度
        for file_name in tqdm(dcm_files):
            # 读取DICOM文件
            full_file_path = os.path.join(file_path, file_name)
            ds = pydicom.dcmread(full_file_path)

            # 读取DICOM信息
            img = ds.pixel_array
            window_center = ds[0x0028, 0x1050].value  # 窗位
            window_width = ds[0x0028, 0x1051].value  # 窗宽
            intercept = ds[0x0028, 0x1052].value  # 截距
            slope = ds[0x0028, 0x1053].value  # 斜率

            # DICOM归一化
            norm_img = normalize_dicom(
                img, window_center, window_width, intercept, slope)

            # 将归一化后的图像添加到列表中
            all_norm_imgs.append(norm_img)

        # 将所有归一化后的图像合并为一个数组并存储为npy文件
        all_norm_imgs = np.stack(all_norm_imgs)
        np.save(output_path, all_norm_imgs)
        print(f"Images from {dose_dir} have been saved to {output_path}")