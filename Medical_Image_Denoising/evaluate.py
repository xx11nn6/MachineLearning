# evaluate.py

import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm


def evaluate_ldct(save_dir, dataset='mayo'):
    """
    评估低剂量CT和原图之间的PSNR和SSIM，并保存结果。
    参数：
    - save_dir: 保存处理后数据的路径
    - dataset: 数据集名称，'mayo' 或 'piglet'
    """
    if dataset == 'mayo':
        full_dose = np.load(os.path.join(
            save_dir, 'mayo_full_dose.npy')).astype(np.float32)
        low_dose = np.load(os.path.join(
            save_dir, 'mayo_low_dose.npy')).astype(np.float32)
    elif dataset == 'piglet':
        full_dose = np.load(os.path.join(
            save_dir, 'piglet_full_dose.npy')).astype(np.float32)
        low_dose = np.load(os.path.join(
            save_dir, 'piglet_low_dose.npy')).astype(np.float32)
    else:
        raise ValueError("数据集参数错误，请选择 'mayo' 或 'piglet'。")

    num_images = full_dose.shape[0]
    ldct_psnr = []
    ldct_ssim = []

    data_range = full_dose.max() - full_dose.min()

    for i in tqdm(range(num_images)):
        gt = full_dose[i]
        ld = low_dose[i]

        ldct_psnr.append(psnr(gt, ld, data_range=data_range))
        ldct_ssim.append(ssim(gt, ld, data_range=data_range))

    # 保存评估结果
    test_info = [ldct_psnr, ldct_ssim]
    np.save(os.path.join(save_dir, f'{dataset}_LDCT_test_info.npy'), test_info)
    print(
        f"低剂量CT评估结果已保存至 {os.path.join(save_dir, f'{dataset}_LDCT_test_info.npy')}")


def evaluate_results(save_dir, dataset='mayo', algorithm='nlm'):
    """
    评估去噪后的结果与原图之间的PSNR和SSIM，并保存结果。
    参数：
    - save_dir: 保存处理后数据的路径
    - dataset: 数据集名称，'mayo' 或 'piglet'
    - algorithm: 算法名称，'nlm'、'bm3d' 或 'redcnn'
    """
    if dataset == 'mayo':
        full_dose = np.load(os.path.join(
            save_dir, 'mayo_full_dose.npy')).astype(np.float32)
        if algorithm == 'nlm':
            denoised = np.load(os.path.join(
                save_dir, 'mayo_nlm_denoised.npy')).astype(np.float32)
        elif algorithm == 'bm3d':
            denoised = np.load(os.path.join(
                save_dir, 'mayo_bm3d_denoised.npy')).astype(np.float32)
        elif algorithm == 'redcnn':
            denoised = np.load(os.path.join(
                save_dir, 'red-cnn-denoised-mayo.npy')).astype(np.float32)
        else:
            raise ValueError("算法参数错误，请选择 'nlm'、'bm3d' 或 'redcnn'。")
    elif dataset == 'piglet':
        full_dose = np.load(os.path.join(
            save_dir, 'piglet_full_dose.npy')).astype(np.float32)
        if algorithm == 'nlm':
            denoised = np.load(os.path.join(
                save_dir, 'piglet_nlm_denoised.npy')).astype(np.float32)
        elif algorithm == 'bm3d':
            denoised = np.load(os.path.join(
                save_dir, 'piglet_bm3d_denoised.npy')).astype(np.float32)
        elif algorithm == 'redcnn':
            denoised = np.load(os.path.join(
                save_dir, 'red-cnn-denoised-piglet.npy')).astype(np.float32)
        else:
            raise ValueError("算法参数错误，请选择 'nlm'、'bm3d' 或 'redcnn'。")
    else:
        raise ValueError("数据集参数错误，请选择 'mayo' 或 'piglet'。")

    num_images = full_dose.shape[0]
    denoised_psnr = []
    denoised_ssim = []

    data_range = full_dose.max() - full_dose.min()

    for i in tqdm(range(num_images)):
        gt = full_dose[i]
        denoised_img = denoised[i]

        denoised_psnr.append(psnr(gt, denoised_img, data_range=data_range))
        denoised_ssim.append(ssim(gt, denoised_img, data_range=data_range))

    # 保存评估结果
    test_info = [denoised_psnr, denoised_ssim]
    np.save(os.path.join(
        save_dir, f'{dataset}_{algorithm}_test_info.npy'), test_info)
    print(f"{algorithm} 算法评估结果已保存至 {os.path.join(save_dir, f'{dataset}_{algorithm}_test_info.npy')}")
