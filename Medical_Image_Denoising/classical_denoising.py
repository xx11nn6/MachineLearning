import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import os
import cv2
import bm3d
from datasets_image_display import zoom_with_psnr_ssim_general


def NLM_opencv(img, h=10, neighbor=7, search=21):
    """
    使用 OpenCV 的 fastNlMeansDenoising 去噪。
    OpenCV 支持 8 位图像，因此需将 16 位图像（范围 0-1）转换为 [0, 255]。
    """
    # 转换为 [0, 255] 的 8 位格式
    img_8bit = (img * 255).astype(np.uint8)
    # 去噪
    denoised = cv2.fastNlMeansDenoising(img_8bit, None, h, neighbor, search)
    # 转回 [0, 1] 的浮点格式
    return denoised.astype(np.float32) / 255.0


def BM3D():
    pass


if __name__ == '__main__':
    current_dir = os.getcwd()
    # 数据集目录
    in_dir = r'dataset\piglet_npy'
    input_names = ['full_dose_ct.npy',
                   'quarter_dose_ct.npy', 'onetenth_dose_ct.npy']
    input_paths = []

    # 遍历每个剂量目录
    for input_name in input_names:
        input_path = os.path.join(in_dir, input_name)
        input_paths.append(input_path)

    full_dose_datasets = np.load(input_paths[0])
    quarter_dose_datasets = np.load(input_paths[1])
    onetenth_dose_datasets = np.load(input_paths[2])

    selected_indices = 319
    # 获取图像
    full_dose_img = full_dose_datasets[selected_indices]
    quarter_dose_img = quarter_dose_datasets[selected_indices]
    onetenth_dose_img = onetenth_dose_datasets[selected_indices]

    # print(result[200][200])
    # print(full_dose_img[200][200])
    result_nlm = NLM_opencv(quarter_dose_img, h=12)
    # bm3d
    result_bm3d = bm3d.bm3d(quarter_dose_img, sigma_psd=0.08,
                            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

    # 将要输出的信息打包
    result_psnr = [None, psnr(full_dose_img, quarter_dose_img), psnr(
        full_dose_img, result_nlm), psnr(full_dose_img, result_bm3d)]
    result_ssim = [None, ssim(full_dose_img, quarter_dose_img, data_range=1), ssim(
        full_dose_img, result_nlm, data_range=1), ssim(full_dose_img, result_bm3d, data_range=1)]
    result = [full_dose_img, quarter_dose_img, result_nlm, result_bm3d]

    # 输出目录
    output_dir = r'images\classical'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果输出目录不存在则创建
    # 输出文件名
    output_names = [f'{selected_indices}_full',
                    f'{selected_indices}_quarter', f'{selected_indices}_nlm', f'{selected_indices}_bm3d']
    for i in range(4):
        savedir = os.path.join(output_dir, output_names[i])
        zoom_with_psnr_ssim_general(result[i], [200, 250], [
                                    100, 150], result_psnr[i], result_ssim[i], savedir)

    plt.figure()
    plt.subplot(141)
    plt.imshow(full_dose_img, cmap='gray')
    plt.subplot(142)
    plt.imshow(quarter_dose_img, cmap='gray')
    plt.subplot(143)
    plt.imshow(result_nlm, cmap='gray')
    plt.subplot(144)
    plt.imshow(result_bm3d, cmap='gray')
    plt.show()
    print(psnr(full_dose_img, quarter_dose_img), psnr(
        full_dose_img, result_nlm), psnr(full_dose_img, result_bm3d))
    print(ssim(full_dose_img, quarter_dose_img, data_range=1), ssim(
        full_dose_img, result_nlm, data_range=1), ssim(full_dose_img, result_bm3d, data_range=1))
