import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from os import makedirs
from os.path import join, isdir
from tqdm.auto import tqdm
from cycleGAN_train import Generator, make_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Functions for caculating PSNR, SSIM
# Peak Signal-to-Noise Ratio


def psnr(A, ref):
    # ref[ref < -1000] = -1000
    # A[A < -1000] = -1000
    # val_min = -1000
    # val_max = np.amax(ref)
    # ref = (ref - val_min) / (val_max - val_min)
    # A = (A - val_min) / (val_max - val_min)
    # print(ref.shape,A.shape)
    out = peak_signal_noise_ratio(ref, A)

    return out.astype(np.float32)

# Structural similarity index


def ssim(A, ref):
    # ref[ref < -1000] = -1000
    # A[A < -1000] = -1000
    # val_min = -1000
    # val_max = np.amax(ref)
    # ref = (ref - val_min) / (val_max - val_min)
    # A = (A - val_min) / (val_max - val_min)
    # 确保输入是2D数组
    if A.ndim == 3:
        A = A[0]  # 如果是3D数组，取第一个通道
    if ref.ndim == 3:
        ref = ref[0]
    out = structural_similarity(ref, A, data_range=2)
    return out

# Test function


def test(
    path_checkpoint='./save_model',
    model_name='cyclegan_v1',
    path_data='../data/AAPM_data',
    path_result='../save',
    dataset='mayo',
    g_channels=32,
    ch_mult=[1, 2, 4, 8],
    num_res_blocks=3,
    num_visualize=6
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path for saving checkpoint
    if not isdir(path_checkpoint):
        makedirs(path_checkpoint)

    # Path for saving model
    path_model = join(path_checkpoint, model_name)
    if not isdir(path_result):
        makedirs(path_result)

    test_dataloader = make_dataloader(path_data, dataset, is_train=False)

    # Load the last checkpoint
    G_Q2F = Generator(1, 1, g_channels, ch_mult=ch_mult,
                      num_res_blocks=num_res_blocks).to(device)
    checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'))
    G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
    G_Q2F.eval()

    # Initialize lists for PSNR and SSIM
    psnr_low = []
    ssim_low = []
    psnr_output = []
    ssim_output = []

    output_images = []
    # Test and calculate metrics
    with torch.no_grad():
        for x_F, x_Q, file_name in tqdm(test_dataloader):
            # 确保数据维度正确
            x_F = x_F.squeeze().numpy()  # 移除批次维度
            x_Q = x_Q.to(device, dtype=torch.float32)
            x_QF = G_Q2F(x_Q)[0].detach().cpu().numpy()
            x_Q = x_Q.squeeze().cpu().numpy()  # 移除批次维度

            # 确保所有图像都是2D的
            if x_Q.ndim == 3:
                x_Q = x_Q.squeeze()
            if x_F.ndim == 3:
                x_F = x_F.squeeze()
            if x_QF.ndim == 3:
                x_QF = x_QF.squeeze()

            # Calculate metrics
            psnr_low.append(psnr(x_Q, x_F))
            ssim_low.append(ssim(x_Q, x_F))
            psnr_output.append(psnr(x_QF, x_F))
            ssim_output.append(ssim(x_QF, x_F))

            # Save output
            output_images.append(x_QF)
            # np.save(join(path_result, f"{file_name[0]}.npy"), x_QF)
    output_images = np.array(output_images, dtype=np.float16)
    np.save(join(path_result, 'piglet_cyclegan_denoised.npy' if dataset ==
            'piglet' else 'mayo_cyclegan_denoised.npy'), output_images)
    print('Output saved')

    print('PSNR and SSIM')
    print('Mean PSNR between input and ground truth:')
    print(np.mean(psnr_low))
    print('Mean SSIM between input and ground truth:')
    print(np.mean(ssim_low))
    print('Mean PSNR between network output and ground truth:')
    print(np.mean(psnr_output))
    print('Mean SSIM between network output and ground truth:')
    print(np.mean(ssim_output))

    # Visualization
    # plt.figure(figsize=(15, 30))
    # num_visualize = 6

    # if dataset == 'mayo':
    #     # For mayo dataset, visualize some random L506 cases
    #     test_files = [f for f in test_dataloader.dataset.file_list]
    #     sampled_indices = random.sample(range(len(test_files)), num_visualize)
    # else:  # piglet dataset
    #     total_images = len(test_dataloader.dataset)
    #     sampled_indices = random.sample(range(total_images), num_visualize)

    # for i, idx in enumerate(sampled_indices):
    #     if dataset == 'mayo':
    #         base_name = test_files[idx]
    #         x_Q = np.load(join(path_data, base_name + '_input.npy'))
    #         x_F = np.load(join(path_data, base_name + '_target.npy'))
    #         x_QF = np.load(join(path_model, f"{base_name}.npy"))
    #     else:  # piglet dataset
    #         x_Q = test_dataloader.dataset.low_dose[idx]
    #         x_F = test_dataloader.dataset.full_dose[idx]
    #         x_QF = np.load(join(path_model, f"{idx}.npy"))

    #     plt.subplot(num_visualize, 3, i * 3 + 1)
    #     plt.imshow(x_Q, cmap='gray')
    #     plt.title(f'low Dose', fontsize=20)
    #     if dataset == 'mayo':
    #         plt.text(-x_Q.shape[1] * 0.1, x_Q.shape[0] * 0.5,
    #                  f'Case: {base_name}', fontsize=20, rotation=90, va='center', ha='center')
    #     else:
    #         plt.text(-x_Q.shape[1] * 0.1, x_Q.shape[0] * 0.5,
    #                  f'Index: {idx}', fontsize=20, rotation=90, va='center', ha='center')
    #     plt.axis('off')

    #     plt.subplot(num_visualize, 3, i * 3 + 2)
    #     plt.imshow(x_F, cmap='gray')
    #     plt.title(f'Full Dose', fontsize=20)
    #     plt.axis('off')

    #     plt.subplot(num_visualize, 3, i * 3 + 3)
    #     plt.imshow(x_QF, cmap='gray')
    #     plt.title(f'Output', fontsize=20)
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.savefig(join(path_model, 'qualitative_results.png'))
    # plt.close()


if __name__ == '__main__':
    # Parse arguments
    # python cycleGAN_test.py --path_data='../save' --path_result='../save' --dataset='mayo'
    # python cycleGAN_test.py --path_data='../dataset/piglet_npy' --path_result='../save' --dataset='piglet'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_checkpoint', type=str,
                        default='./save_model')
    parser.add_argument('--model_name', type=str, default='cyclegan_v1')
    parser.add_argument('--path_data', type=str, default='./AAPM_data')
    parser.add_argument('--path_result', type=str, default='../save')
    parser.add_argument('--g_channels', type=int, default=32)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_visualize', type=int, default=6)
    parser.add_argument('--dataset', type=str, choices=['piglet', 'mayo'], default='mayo',
                        help='Dataset to use (piglet or mayo)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    test(
        path_checkpoint=args.path_checkpoint,
        model_name=args.model_name,
        path_data=args.path_data,
        path_result=args.path_result,
        dataset=args.dataset,
        g_channels=args.g_channels,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
        num_visualize=args.num_visualize
    )
