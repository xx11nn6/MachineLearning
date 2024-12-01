# show_test_info.py

import numpy as np
import matplotlib.pyplot as plt
import os


def display_results(save_dir, dataset='mayo', algorithm='nlm'):
    """
    显示评估结果的曲线。
    参数：
    - save_dir: 评估结果的保存路径
    - dataset: 数据集名称，'mayo' 或 'piglet'
    - algorithm: 算法名称，'LDCT'、'nlm'、'bm3d' 或 'redcnn'
    """
    if algorithm == 'LDCT':
        # 加载低剂量CT的评估结果
        test_info = np.load(os.path.join(
            save_dir, f'{dataset}_LDCT_test_info.npy'), allow_pickle=True)
        psnr_values, ssim_values = test_info

        # 绘制 PSNR 曲线
        plt.figure(figsize=(10, 5))
        plt.plot(psnr_values, label='Low Dose')
        plt.title(f'{dataset.upper()} - Low Dose PSNR')
        plt.xlim(0, len(psnr_values) - 1)
        plt.xlabel('Slice Index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)

        # 添加均值的文字描述
        plt.text(
            0.5, 0.9, f'Low Dose Mean PSNR: {np.mean(psnr_values):.4f}', transform=plt.gca().transAxes)

        # 保存 PSNR 曲线图
        plt.savefig(os.path.join(save_dir, f'{dataset}_LDCT_psnr_curve.png'))
        plt.show()

        # 绘制 SSIM 曲线
        plt.figure(figsize=(10, 5))
        plt.plot(ssim_values, label='Low Dose')
        plt.title(f'{dataset.upper()} - Low Dose SSIM')
        plt.xlim(0, len(ssim_values) - 1)
        plt.xlabel('Slice Index')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True)

        # 添加均值的文字描述
        plt.text(
            0.5, 0.9, f'Low Dose Mean SSIM: {np.mean(ssim_values):.4f}', transform=plt.gca().transAxes)

        # 保存 SSIM 曲线图
        plt.savefig(os.path.join(save_dir, f'{dataset}_LDCT_ssim_curve.png'))
        plt.show()
    else:
        # 加载去噪后图像的评估结果
        test_info = np.load(os.path.join(
            save_dir, f'{dataset}_{algorithm}_test_info.npy'), allow_pickle=True)
        denoised_psnr, denoised_ssim = test_info

        # 绘制 PSNR 曲线
        plt.figure(figsize=(10, 5))
        plt.plot(denoised_psnr, label=algorithm.upper())
        plt.title(f'{dataset.upper()} - {algorithm.upper()} PSNR')
        plt.xlim(0, len(denoised_psnr) - 1)
        plt.xlabel('Slice Index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)

        # 添加均值的文字描述
        plt.text(0.5, 0.9, f'{algorithm.upper()} Mean PSNR: {np.mean(denoised_psnr):.4f}',
                 transform=plt.gca().transAxes)

        # 保存 PSNR 曲线图
        plt.savefig(os.path.join(
            save_dir, f'{dataset}_{algorithm}_psnr_curve.png'))
        plt.show()

        # 绘制 SSIM 曲线
        plt.figure(figsize=(10, 5))
        plt.plot(denoised_ssim, label=algorithm.upper())
        plt.title(f'{dataset.upper()} - {algorithm.upper()} SSIM')
        plt.xlim(0, len(denoised_ssim) - 1)
        plt.xlabel('Slice Index')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True)

        # 添加均值的文字描述
        plt.text(0.5, 0.9, f'{algorithm.upper()} Mean SSIM: {np.mean(denoised_ssim):.4f}',
                 transform=plt.gca().transAxes)

        # 保存 SSIM 曲线图
        plt.savefig(os.path.join(
            save_dir, f'{dataset}_{algorithm}_ssim_curve.png'))
        plt.show()
