import os
import argparse
import numpy as np
from classical_denoising import process_mayo_data, process_piglet_data
from evaluate import evaluate_results, evaluate_ldct
from show_test_info import display_results
from datasets_image_display import zoom_with_psnr_ssim_general
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"创建保存目录：{args.save_dir}")

    if args.function == 1:
        # 功能 1：使用 NLM 算法处理数据集并保存图像
        if args.algorithm != 'nlm':
            raise ValueError("功能 1 只能选择算法 'nlm'")
        if args.dataset == 'mayo':
            print("使用 NLM 算法处理 Mayo 数据集...")
            process_mayo_data(args.data_dir, args.save_dir, algorithms=['nlm'])
        elif args.dataset == 'piglet':
            print("使用 NLM 算法处理 Piglet 数据集...")
            process_piglet_data(
                args.data_dir, args.save_dir, algorithms=['nlm'])
        else:
            raise ValueError("数据集参数错误，请选择 'mayo' 或 'piglet'。")
    elif args.function == 2:
        # 功能 2：使用 BM3D 算法处理数据集并保存图像
        if args.algorithm != 'bm3d':
            raise ValueError("功能 2 只能选择算法 'bm3d'")
        if args.dataset == 'mayo':
            print("使用 BM3D 算法处理 Mayo 数据集...")
            process_mayo_data(args.data_dir, args.save_dir,
                              algorithms=['bm3d'])
        elif args.dataset == 'piglet':
            print("使用 BM3D 算法处理 Piglet 数据集...")
            process_piglet_data(
                args.data_dir, args.save_dir, algorithms=['bm3d'])
        else:
            raise ValueError("数据集参数错误，请选择 'mayo' 或 'piglet'。")
    elif args.function == 3:
        # 功能 3：评估选定的文件
        if args.algorithm == 'LDCT':
            # 评估低剂量CT
            print(f"评估低剂量CT在 {args.dataset} 数据集上的结果...")
            evaluate_ldct(args.save_dir, dataset=args.dataset)
        else:
            # 评估去噪后图像
            print(f"评估 {args.algorithm} 算法在 {args.dataset} 数据集上的结果...")
            evaluate_results(args.save_dir, dataset=args.dataset,
                             algorithm=args.algorithm)

    elif args.function == 4:
        # 功能 4：显示整体测试效果
        print(f"显示不同算法在 {args.dataset} 数据集上的测试效果...")
        display_results(args.save_dir, dataset=args.dataset)

    elif args.function == 5:
        # 功能 5：显示并保存某一测试集中指定图像的局部放大、不同算法的对比图
        print('局部放大图')
        index = args.index  # 从命令行参数获取图像索引
        save_dir = args.save_dir
        dataset = args.dataset
        zoom_coords = args.zoom_coords  # 从命令行参数获取局部放大坐标，例如 "x1,x2,y1,y2"

        # 解析 zoom_coords 参数，得到 [x1, x2] 和 [y1, y2]
        zoom_coords_list = [int(coord) for coord in zoom_coords.split(',')]
        x = [zoom_coords_list[0], zoom_coords_list[1]]
        y = [zoom_coords_list[2], zoom_coords_list[3]]
        enum_img = ['full_dose', 'low_dose', 'nlm_denoised',
                    'bm3d_denoised', 'redcnn_denoised', 'cyclegan_denoised']
        enum_info = ['LDCT', 'nlm', 'bm3d', 'redcnn', 'cyclegan']
        images = []
        psnr_all = []
        ssim_all = []

        # 获取图像
        image_savepath = './images'
        for i, algo in enumerate(enum_img):
            image = np.load(os.path.join(
                save_dir, f'{dataset}_{algo}.npy'), allow_pickle=True)
            images.append(image[index])
        images = np.array(images)
        # 获取PSNR,SSIM
        for i, algo in enumerate(enum_info):
            info = np.load(os.path.join(
                save_dir, f'{dataset}_{algo}_test_info.npy'), allow_pickle=True)
            p, s = info
            psnr_all.append(p[index])
            ssim_all.append(s[index])
        # 绘图及保存
        plt.figure(figsize=(12, 5))
        for i, algo in enumerate(enum_img):
            plt.subplot(1, 6, i+1)
            if i == 0:
                result = zoom_with_psnr_ssim_general(images[i], x, y, None, None, os.path.join(
                    image_savepath, f'{dataset}_{index}_full_dose.png'))
                plt.imshow(result)
                plt.title('Full Dose')
            else:
                result = zoom_with_psnr_ssim_general(
                    images[i], x, y, psnr_all[i-1], ssim_all[i-1], os.path.join(
                        image_savepath, f'{dataset}_{index}_{algo}.png'))
                plt.imshow(result)
                plt.title(f'{algo}')
        plt.xlabel(f'{dataset}')
        plt.axis('off')
        plt.show()
        # for i in range(len(enum_img)):

        # zoom_with_psnr_ssim_general
        print(images.shape)


if __name__ == '__main__':
    # 功能 1：使用 NLM 算法处理数据集并保存图像
    # python main.py --function=1 --dataset=mayo --data_dir=./save --save_dir=./save --algorithm=nlm
    # python main.py --function=1 --dataset=piglet --data_dir=./dataset/piglet_npy --save_dir=./save --algorithm=nlm
    # 功能 2：使用 BM3D 算法处理数据集并保存图像
    # python main.py --function=2 --dataset=mayo --data_dir=./save --save_dir=./save --algorithm=bm3d
    # 功能 3：评估选定的文件
    # 评估低剂量CT
    # python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=LDCT
    # nlm
    # python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=nlm
    # 功能 4：显示整体测试效果图像
    # python main.py --function=4 --dataset=mayo --save_dir=./save
    # 功能 5：显示并保存某一测试集中指定图像的局部放大、不同算法的对比图
    # python main.py --function=5 --dataset=mayo --save_dir=./save --zoom_coords='200,250,200,250' --index=365
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=int, choices=[1, 2, 3, 4, 5], required=True,
                        help="选择功能：1-使用 NLM 算法处理并保存图像，2-使用 BM3D 算法处理并保存图像，3-评估选定的文件，4-显示整体测试效果，5-显示并保存指定图像的局部放大和算法对比图")
    parser.add_argument('--dataset', type=str, choices=['mayo', 'piglet'], required=True,
                        help="选择数据集，取值为 'mayo' 或 'piglet'")
    parser.add_argument('--data_dir', type=str,
                        help="数据集路径")
    parser.add_argument('--save_dir', type=str, default='./save',
                        help="保存处理后数据的路径")
    parser.add_argument('--algorithm', type=str, choices=['nlm', 'bm3d', 'redcnn', 'cyclegan', 'LDCT'], required=False,
                        help="选择算法，取值为 'nlm'、'bm3d' 或 'redcnn'")
    parser.add_argument('--index', type=int, default=365,
                        help="指定图像的索引，用于功能 5")
    parser.add_argument('--zoom_coords', type=str, default='200,250,200,250',
                        help="指定局部放大坐标，格式为 'x1,x2,y1,y2'，用于功能5")
    args = parser.parse_args()
    main(args)
