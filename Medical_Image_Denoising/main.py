import os
import argparse
import numpy as np
from classical_denoising import process_mayo_data, process_piglet_data
from evaluate import evaluate_results, evaluate_ldct
from show_test_info import display_results
from datasets_image_display import zoom_with_psnr_ssim_general
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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
        print(f"显示 {args.algorithm} 在 {args.dataset} 数据集上的测试效果...")
        display_results(args.save_dir, dataset=args.dataset,
                        algorithm=args.algorithm)

    elif args.function == 5:
        # 功能 5：显示并保存某一测试集中指定图像的局部放大、不同算法的对比图
        index = args.index  # 从命令行参数获取图像索引
        zoom_coords = args.zoom_coords  # 从命令行参数获取局部放大坐标，例如 "x1,x2,y1,y2"

        # 解析 zoom_coords 参数，得到 [x1, x2] 和 [y1, y2]
        zoom_coords_list = [int(coord) for coord in zoom_coords.split(',')]
        if len(zoom_coords_list) != 4:
            raise ValueError("zoom_coords 应该包含四个整数，格式为 'x1,x2,y1,y2'")
        zoom_x = zoom_coords_list[0:2]
        zoom_y = zoom_coords_list[2:4]

        print(f"显示并保存第 {index} 张图像的局部放大和算法对比图...")

        # 加载图像
        if args.dataset == 'mayo':
            full_dose = np.load(os.path.join(
                args.save_dir, 'mayo_full_dose.npy')).astype(np.float32)
            low_dose = np.load(os.path.join(
                args.save_dir, 'mayo_low_dose.npy')).astype(np.float32)
            nlm_path = os.path.join(args.save_dir, 'mayo_nlm_denoised.npy')
            bm3d_path = os.path.join(args.save_dir, 'mayo_bm3d_denoised.npy')
            redcnn_path = os.path.join(
                args.save_dir, 'red-cnn-denoised-mayo.npy')
        elif args.dataset == 'piglet':
            full_dose = np.load(os.path.join(
                args.save_dir, 'piglet_full_dose.npy')).astype(np.float32)
            low_dose = np.load(os.path.join(
                args.save_dir, 'piglet_low_dose.npy')).astype(np.float32)
            nlm_path = os.path.join(args.save_dir, 'piglet_nlm_denoised.npy')
            bm3d_path = os.path.join(args.save_dir, 'piglet_bm3d_denoised.npy')
            redcnn_path = os.path.join(
                args.save_dir, 'red-cnn-denoised-piglet.npy')
        else:
            raise ValueError("数据集参数错误，请选择 'mayo' 或 'piglet'。")

        # 获取指定索引的图像
        gt_img = full_dose[index]
        ld_img = low_dose[index]

        # 初始化字典，存储图像和指标
        images_dict = {'Ground Truth': (gt_img, Nr, None),  # PSNR 和 SSIM 为 None
                       'Low Dose': (ld_img, None, None)}       # 待计算

        # 如果存在 NLM 结果，加载并存储
        if os.path.exists(nlm_path):
            nlm_denoised = np.load(nlm_path).astype(np.float32)
            nlm_img = nlm_denoised[index]
            images_dict['NLM'] = (nlm_img, None, None)  # 待计算

        # 如果存在 BM3D 结果，加载并存储
        if os.path.exists(bm3d_path):
            bm3d_denoised = np.load(bm3d_path).astype(np.float32)
            bm3d_img = bm3d_denoised[index]
            images_dict['BM3D'] = (bm3d_img, None, None)  # 待计算

        # 如果存在 RED-CNN 结果，加载并存储
        if os.path.exists(redcnn_path):
            redcnn_denoised = np.load(redcnn_path).astype(np.float32)
            redcnn_img = redcnn_denoised[index]
            images_dict['RED-CNN'] = (redcnn_img, None, None)  # 待计算

        # 计算 PSNR 和 SSIM
        data_range = gt_img.max() - gt_img.min()
        for name, (img, _, _) in images_dict.items():
            if name != 'Ground Truth':
                psnr_value = psnr(gt_img, img, data_range=data_range)
                ssim_value = ssim(gt_img, img, data_range=data_range)
                images_dict[name] = (img, psnr_value, ssim_value)

        # 调用 zoom_with_psnr_ssim_general 函数，对每个图像进行处理并保存
        output_dir = os.path.join(args.save_dir, 'comparison_images')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for name, (img, psnr_value, ssim_value) in images_dict.items():
            output_path = os.path.join(
                output_dir, f"{args.dataset}_{index}_{name}.png")
            zoom_with_psnr_ssim_general(
                img, zoom_x, zoom_y, psnr_value, ssim_value, output_path)
            print(f"已保存图像：{output_path}")
    else:
        raise ValueError("功能参数错误，请选择 1、2、3、4 或 5。")


if __name__ == '__main__':
    # 功能 1：使用 NLM 算法处理数据集并保存图像
    # python main.py --function=1 --dataset=mayo --data_dir=./dataset/mayo_npy --save_dir=./save --algorithm=nlm
    # python main.py --function=1 --dataset=piglet --data_dir=./dataset/piglet_npy --save_dir=./save --algorithm=nlm
    # 功能 2：使用 BM3D 算法处理数据集并保存图像
    # python main.py --function=2 --dataset=mayo --data_dir=./dataset/mayo_npy --save_dir=./save --algorithm=bm3d
    # 功能 3：评估选定的文件
    # 评估低剂量CT
    # python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=LDCT
    # nlm
    # python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=nlm
    # 功能 4：显示整体测试效果图像
    # python main.py --function=4 --dataset=mayo --save_dir=./save --algorithm=nlm
    # 功能 5：显示并保存某一测试集中指定图像的局部放大、不同算法的对比图
    # python main.py --function=5 --dataset=mayo --save_dir=./save --index=10
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=int, choices=[1, 2, 3, 4, 5], required=True,
                        help="选择功能：1-使用 NLM 算法处理并保存图像，2-使用 BM3D 算法处理并保存图像，3-评估选定的文件，4-显示整体测试效果，5-显示并保存指定图像的局部放大和算法对比图")
    parser.add_argument('--dataset', type=str, choices=['mayo', 'piglet'], required=True,
                        help="选择数据集，取值为 'mayo' 或 'piglet'")
    parser.add_argument('--data_dir', type=str,
                        help="数据集路径")
    parser.add_argument('--save_dir', type=str, default='./save',
                        help="保存处理后数据的路径")
    parser.add_argument('--algorithm', type=str, choices=['nlm', 'bm3d', 'redcnn', 'LDCT'], required=False,
                        help="选择算法，取值为 'nlm'、'bm3d' 或 'redcnn'")
    parser.add_argument('--index', type=int, default=0,
                        help="指定图像的索引，用于功能 5")
    parser.add_argument('--zoom_coords', type=str, default='200,250,100,150',
                        help="指定局部放大坐标，格式为 'x1,x2,y1,y2'，用于功能5")
    args = parser.parse_args()
    main(args)
