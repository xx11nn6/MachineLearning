from measure import compute_measure
from networks import RED_CNN
from prep import printProgressBar
import torch.optim as optim
import torch.nn as nn
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        self.REDCNN = RED_CNN()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)

    def save_model(self, iter_):
        # 用于保存模型的状态字典（state_dict），保存的文件名包含迭代次数。
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)

    def load_model(self, iter_):
        # 加载模型的状态字典。若使用多 GPU，需要处理 DataParallel 生成的 module. 前缀。若不使用多 GPU，直接加载模型。
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))

    def lr_decay(self):
        # 每当调用这个方法时，学习率会衰减为原来的一半。用于控制训练过程中的学习率衰减。
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):
        # 将归一化的图像数据恢复到原始的数值范围
        image = image * (self.norm_range_max -
                         self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        # 对图像进行裁剪，确保所有值都在指定的范围 [trunc_min, trunc_max] 内
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        # x, y, pred 应该是 numpy 数组
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(
            original_result[0], original_result[1], original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(
            pred_result[0], pred_result[1], pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig',
                  'result_{}.png'.format(fig_name)))
        plt.close()

    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            self.REDCNN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch_size:  # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.REDCNN(x)
                loss = self.criterion(pred, y)
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch,
                                                                                                        self.num_epochs, iter_+1,
                                                                                                        len(self.data_loader), loss.item(
                                                                                                        ),
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(
                        total_iters)), np.array(train_losses))

    def test(self):
        # 评估模型在测试集上的表现
        del self.REDCNN
        # load
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        # 新增：用于存储去噪后的图像
        denoised_images = []
        full_dose = []
        low_dose = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.REDCNN(x)
                pred = pred.view(shape_, shape_).cpu().detach()
                x = x.view(shape_, shape_).cpu().detach()
                y = y.view(shape_, shape_).cpu().detach()
                # denormalize, truncate
                x_denorm = self.trunc(self.denormalize_(x))
                y_denorm = self.trunc(self.denormalize_(y))
                pred_denorm = self.trunc(self.denormalize_(pred))

                # 存储处理后的图像，转换为 numpy 数组并添加到列表中
                denoised_images.append(pred.numpy().astype(np.float32))
                full_dose.append(y.numpy().astype(np.float32))
                low_dose.append(x.numpy().astype(np.float32))
                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(
                    x_denorm, y_denorm, pred_denorm, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                # if self.result_fig:
                #     self.save_fig(x_denorm, y_denorm, pred_denorm,
                #                   i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.data_loader),
                ori_ssim_avg / len(self.data_loader),
                ori_rmse_avg / len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.data_loader),
                pred_ssim_avg / len(self.data_loader),
                pred_rmse_avg / len(self.data_loader)))

        # 保存处理后的图像
        denoised_images = np.array(denoised_images)  # 形状为 (n, height, width)
        full_dose = np.array(full_dose)
        low_dose = np.array(low_dose)

        output_dir = '../save'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(os.path.join(output_dir, 'mayo_redcnn_denoised.npy'),
                denoised_images.astype(np.float16))
        np.save(os.path.join(output_dir, 'mayo_full_dose.npy'),
                full_dose.astype(np.float16))
        np.save(os.path.join(output_dir, 'mayo_low_dose.npy'),
                low_dose.astype(np.float16))

        print(f'去噪后的图像已保存至 {output_dir}')

    def test_piglet(self):
        # 加载模型
        del self.REDCNN
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_iters)

        # 评估指标初始化
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        # **新增：用于存储去噪后的图像**
        denoised_images = []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                x = x.float().to(self.device)  # x 的形状为 [batch_size, H, W]
                y = y.float().to(self.device)

                # 添加通道维度
                x = x.unsqueeze(1)  # 形状变为 [batch_size, 1, H, W]
                y = y.unsqueeze(1)

                pred = self.REDCNN(x)

                data_range = 1.0  # 数据范围为 0-1

                # 将张量转换为 NumPy 数组
                x_np = x.squeeze().cpu().numpy()
                y_np = y.squeeze().cpu().numpy()
                pred_np = pred.squeeze().cpu().numpy()

                # **确保形状为 (512, 512)，并添加新的维度 (1, 512, 512)**
                pred_np = pred_np.squeeze()  # 移除多余的维度，变为 (512, 512)
                # 添加到列表中，保持形状为 (1, 512, 512)
                denoised_images.append(pred_np[np.newaxis, ...])

                # 调用 compute_measure 函数

                original_result, pred_result = compute_measure(
                    x_np, y_np, pred_np, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # 保存结果图像
                if self.result_fig:
                    self.save_fig(x_np, y_np, pred_np, i,
                                  original_result, pred_result)

                printProgressBar(i + 1, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            num = len(self.data_loader)
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / num, ori_ssim_avg / num, ori_rmse_avg / num))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / num, pred_ssim_avg / num, pred_rmse_avg / num))

            # **在测试结束后，保存处理后的所有图像**
            denoised_images = np.concatenate(
                # 形状为 (n, 512, 512)
                denoised_images, axis=0).astype(np.float32)
            # 确保输出目录存在
            output_dir = '../save'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存为 .npy 文件
            output_path = os.path.join(
                output_dir, 'piglet_redcnn_denoised.npy')
            np.save(output_path, denoised_images.astype(np.float16))
            print(f'去噪后的图像已保存至 {output_path}')
