import os
import argparse
from torch.backends import cudnn
from loader import get_loader, get_loader_piglet
from solver import Solver


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    if args.mode == 'train':
        if args.dataset == 'mayo':
            # 使用 Mayo 数据集进行训练
            data_loader = get_loader(mode='train',
                                     load_mode=args.load_mode,
                                     saved_path=args.path_data,
                                     test_patient='L506',
                                     patch_n=args.patch_n,
                                     patch_size=args.patch_size,
                                     transform=args.transform,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)
            solver = Solver(args, data_loader)
            solver.train()
        else:
            raise ValueError("训练模式下仅支持 'mayo' 数据集")
    elif args.mode == 'test':
        if args.dataset == 'mayo':
            # 使用 Mayo 数据集进行测试（L506 患者）
            data_loader = get_loader(mode='test',
                                     load_mode=args.load_mode,
                                     saved_path=args.path_data,
                                     test_patient='L506',
                                     transform=args.transform,
                                     batch_size=1,
                                     num_workers=args.num_workers)
            solver = Solver(args, data_loader)
            solver.test()
        elif args.dataset == 'piglet':
            # 使用 Piglet 数据集进行测试
            data_loader = get_loader_piglet(data_path=args.path_data,
                                            batch_size=1,
                                            num_workers=args.num_workers)
            solver = Solver(args, data_loader)
            solver.test_piglet()
        else:
            raise ValueError("不支持的数据集: {}".format(args.dataset))
    else:
        raise ValueError("不支持的模式: {}".format(args.mode))


if __name__ == "__main__":
    # python main.py --mode=train --load_mode=0 --batch_size=16 --num_epochs=100 --lr=1e-5 --device='cuda' --num_workers=7
    # python main.py --mode='test' --dataset='piglet' --path_data='../dataset/piglet_npy' --test_iters=33000
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)

    # 在 argparse 部分添加新参数
    parser.add_argument('--dataset', type=str, choices=['piglet', 'mayo'], default='mayo',
                        help="选择数据集，取值为 'piglet' 或 'mayo'")
    parser.add_argument('--path_data', type=str, default='../dataset/mayo_npy',
                        help="数据集的路径")

    parser.add_argument('--data_path', type=str, default='../dataset/mayo')
    parser.add_argument('--saved_path', type=str,
                        default='../dataset/mayo_npy')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
