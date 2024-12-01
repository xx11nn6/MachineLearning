import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(save_path):
    """
    从指定的保存路径中读取损失值文件，按照指定方式计算每个 epoch 的损失值，并绘制损失曲线。

    参数：
    - save_path: 保存模型和损失值的路径
    """
    # 获取保存路径中的所有文件
    files = os.listdir(save_path)

    # 筛选出损失值文件，文件名格式为 'loss_{total_iters}_iter.npy'
    loss_files = [f for f in files if f.startswith(
        'loss_') and f.endswith('_iter.npy')]

    if not loss_files:
        print("在指定的保存路径中未找到损失值文件。")
        return

    # 按照迭代次数排序损失值文件
    loss_files.sort(key=lambda x: int(x.split('_')[1]))

    # 初始化一个列表，用于存储所有的损失值
    all_losses = []

    # 逐个读取损失值文件，并将损失值添加到列表中
    for loss_file in loss_files:
        loss_path = os.path.join(save_path, loss_file)
        losses = np.load(loss_path)
        all_losses.extend(losses)

    # 将损失值转换为 NumPy 数组
    all_losses = np.array(all_losses)

    # 总迭代次数
    total_iters = len(all_losses)
    print(f"总迭代次数：{total_iters}")

    # 定义每个 epoch 的损失值列表
    epoch_losses = []

    # 第 1 个 epoch，损失值为第 1 次迭代的损失值
    epoch_losses.append(all_losses[0])

    # 总的 epoch 数量
    num_epochs = 100
    print(f"总共的 epoch 数量：{num_epochs}")

    # 计算剩余的 epoch 数量
    remaining_epochs = num_epochs - 1

    # 从第 2 个 epoch 开始
    for epoch_idx in range(1, num_epochs):
        # 计算当前 epoch 对应的迭代范围
        # 起始迭代索引，从第 1000 次迭代开始，每 1000 次迭代取一次
        start_iter = epoch_idx * 1000
        end_iter = start_iter + 333

        # 确保不超过总的迭代次数
        if start_iter >= total_iters:
            print(f"迭代次数不足，无法计算第 {epoch_idx+1} 个 epoch 的损失。")
            break
        if end_iter > total_iters:
            end_iter = total_iters

        # 计算该 epoch 的损失均值
        epoch_loss = np.mean(all_losses[start_iter:end_iter])
        epoch_losses.append(epoch_loss)

    # 实际计算的 epoch 数量
    actual_epochs = len(epoch_losses)
    print(f"实际计算的 epoch 数量：{actual_epochs}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 保存损失曲线图像
    loss_curve_path = os.path.join(save_path, 'loss_curve_epochs.png')
    plt.savefig(loss_curve_path)
    print(f"损失曲线已保存至 {loss_curve_path}")

    # 显示损失曲线
    plt.show()


if __name__ == '__main__':
    # 指定保存路径，这里假设与 solver.py 中的 self.save_path 相同
    save_path = './save'  # 请根据实际情况修改路径

    plot_loss_curve(save_path)
