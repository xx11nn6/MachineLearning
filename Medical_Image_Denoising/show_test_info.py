# show_test_info.py

import numpy as np
import matplotlib.pyplot as plt
import os


def display_results(save_dir, dataset='mayo'):
    """
    显示评估结果的曲线。
    参数：
    - save_dir: 评估结果的保存路径
    - dataset: 数据集名称，'mayo' 或 'piglet'
    - algorithm: 算法名称，'LDCT'、'nlm'、'bm3d' 或 'redcnn'
    """
    enum = ['LDCT', 'nlm', 'bm3d', 'redcnn', 'cyclegan']
    psnr_all = []
    ssim_all = []
    for index, algo in enumerate(enum):
        # print(algo)
        psnr, ssim = np.load(os.path.join(
            save_dir, f'{dataset}_{algo}_test_info.npy'), allow_pickle=True)
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_all[0], label='Low Dose')
    plt.plot(psnr_all[1], label='NLM')
    plt.plot(psnr_all[2], label='BM3D')
    plt.plot(psnr_all[3], label='RED-CNN')
    plt.plot(psnr_all[4], label='CycleGAN')

    plt.title('PSNR')
    plt.xlim(0, len(psnr_all[0]) - 1)
    plt.xlabel(f'{dataset} Slice Index')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid(True)

    # 添加均值的文字描述
    plt.text(
        0.01, 0.21, f'Low Dose Mean PSNR: {np.mean(psnr_all[0]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.16, f'NLM Mean PSNR: {np.mean(psnr_all[1]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.11, f'BM3D Mean PSNR: {np.mean(psnr_all[2]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.06, f'RED-CNN Mean PSNR: {np.mean(psnr_all[3]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.01, f'CycleGAN Mean PSNR: {np.mean(psnr_all[4]):.4f}', transform=plt.gca().transAxes)
    plt.savefig(f'{dataset}_psnr_curve.png')  # 保存PSNR曲线图
    plt.show()

    # 绘制SSIM的曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(ssim_all[0], label='Low Dose')
    plt.plot(ssim_all[1], label='NLM')
    plt.plot(ssim_all[2], label='BM3D')
    plt.plot(ssim_all[3], label='RED-CNN')
    plt.plot(ssim_all[4], label='CycleGAN')
    plt.xlim(0, len(ssim_all[0]) - 1)
    plt.title('SSIM')
    plt.xlabel(f'{dataset} Slice Index')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)

    # 添加均值的文字描述
    plt.text(
        0.01, 0.21, f'Low Dose Mean SSIM: {np.mean(ssim_all[0]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.16, f'NLM Mean SSIM: {np.mean(ssim_all[1]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.11, f'BM3D Mean SSIM: {np.mean(ssim_all[2]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.06, f'RED-CNN Mean SSIM: {np.mean(ssim_all[3]):.4f}', transform=plt.gca().transAxes)
    plt.text(
        0.01, 0.01, f'CycleGAN Mean SSIM: {np.mean(ssim_all[4]):.4f}', transform=plt.gca().transAxes)
    plt.savefig(f'{dataset}_ssim_curve.png')  # 保存SSIM曲线图
    plt.show()


if __name__ == '__main__':
    save_dir = './save'
    display_results(save_dir, dataset='mayo')
