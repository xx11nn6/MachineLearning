import numpy as np
import os
import cv2
import bm3d
from datasets_image_display import zoom_with_psnr_ssim_general
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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


def process_mayo_data(saved_path, save_dir, algorithms=['nlm', 'bm3d']):
    """
    处理 Mayo 数据集中的 L506 患者数据。
    参数：
    - saved_path: Mayo 数据集保存的路径
    - save_dir: 处理后数据的保存路径
    """
    # 获取 L506 患者的数据文件列表
    input_files = sorted([f for f in os.listdir(
        saved_path) if ('L506' in f) and ('_input.npy' in f)])
    target_files = sorted([f for f in os.listdir(
        saved_path) if ('L506' in f) and ('_target.npy' in f)])

    # 初始化数组，保存处理后的数据
    num_images = len(input_files)
    full_dose = np.zeros((num_images, 512, 512), dtype=np.float16)
    low_dose = np.zeros((num_images, 512, 512), dtype=np.float16)
    # 根据选择的算法，初始化对应的数组
    if 'nlm' in algorithms:
        nlm_denoised = np.zeros((num_images, 512, 512), dtype=np.float16)
    if 'bm3d' in algorithms:
        bm3d_denoised = np.zeros((num_images, 512, 512), dtype=np.float16)

    for i in tqdm(range(num_images)):
        # 读取图像
        input_img = np.load(os.path.join(
            saved_path, input_files[i])).astype(np.float32)
        target_img = np.load(os.path.join(
            saved_path, target_files[i])).astype(np.float32)

        if 'nlm' in algorithms:
            # NLM 去噪
            result_nlm = NLM_opencv(input_img, h=12)
            nlm_denoised[i] = result_nlm.astype(np.float16)
        if 'bm3d' in algorithms:
            # BM3D 去噪
            result_bm3d = bm3d.bm3d(input_img, sigma_psd=0.08,
                                    stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING).astype(np.float32)
            bm3d_denoised[i] = result_bm3d.astype(np.float16)

        # 保存图像
        full_dose[i] = target_img.astype(np.float16)
        low_dose[i] = input_img.astype(np.float16)

    # 保存处理后的数据
    np.save(os.path.join(save_dir, 'mayo_full_dose.npy'), full_dose)
    np.save(os.path.join(save_dir, 'mayo_low_dose.npy'), low_dose)
    if 'nlm' in algorithms:
        np.save(os.path.join(save_dir, 'mayo_nlm_denoised.npy'), nlm_denoised)
    if 'bm3d' in algorithms:
        np.save(os.path.join(save_dir, 'mayo_bm3d_denoised.npy'), bm3d_denoised)
    print(f"处理后的数据已保存至 {save_dir}")


def process_piglet_data(data_dir, save_dir, algorithms=['nlm', 'bm3d']):
    """
    处理 Piglet 数据集。
    参数：
    - data_dir: Piglet 数据集的路径
    - save_dir: 处理后数据的保存路径
    """
    # 加载数据集
    full_dose = np.load(os.path.join(
        data_dir, 'full_dose_ct.npy')).astype(np.float16)
    low_dose = np.load(os.path.join(
        data_dir, 'onetenth_dose_ct.npy')).astype(np.float16)

    num_images = full_dose.shape[0]
    # 根据选择的算法，初始化对应的数组
    if 'nlm' in algorithms:
        nlm_denoised = np.zeros((num_images, 512, 512), dtype=np.float16)
    if 'bm3d' in algorithms:
        bm3d_denoised = np.zeros((num_images, 512, 512), dtype=np.float16)

    for i in tqdm(range(num_images)):
        input_img = low_dose[i]

        if 'nlm' in algorithms:
            # NLM 去噪
            result_nlm = NLM_opencv(input_img, h=12)
            nlm_denoised[i] = result_nlm.astype(np.float16)
        if 'bm3d' in algorithms:
            # BM3D 去噪
            result_bm3d = bm3d.bm3d(input_img, sigma_psd=0.08,
                                    stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING).astype(np.float32)
            bm3d_denoised[i] = result_bm3d.astype(np.float16)

    # 保存处理后的数据
    np.save(os.path.join(save_dir, 'piglet_full_dose.npy'),
            full_dose.astype(np.float16))
    np.save(os.path.join(save_dir, 'piglet_low_dose.npy'),
            low_dose.astype(np.float16))
    if 'nlm' in algorithms:
        np.save(os.path.join(save_dir, 'piglet_nlm_denoised.npy'), nlm_denoised)
    if 'bm3d' in algorithms:
        np.save(os.path.join(save_dir, 'piglet_bm3d_denoised.npy'), bm3d_denoised)
    print(f"处理后的数据已保存至 {save_dir}")


if __name__ == '__main__':
    pass  # 此处不做任何操作，由 main.py 调用
