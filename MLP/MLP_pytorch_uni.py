# torch实现MLP，代码为一般pytorch编写神经网络的框架
# 其中每一层、损失函数、优化函数、训练过程直接用torch中已定义的类和方法

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

activation_dict = {
    'identity': lambda x: x,
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'relu': nn.ReLU()
}


class MLP(nn.Module):
    def __init__(self, layer_size, use_bias=True, out_activation='sigmoid'):
        super(MLP, self).__init__()

        # 生成网络
        self.mlp = nn.Sequential()
        # 线性层+ReLU
        for dim_idx in range(len(layer_size)-2):
            self.mlp.add_module(
                f'Linear {dim_idx+1}', nn.Linear(layer_size[dim_idx], layer_size[dim_idx+1], use_bias))
            self.mlp.add_module(f'ReLU {dim_idx+1}', nn.ReLU())
        # 最后一层
        self.mlp.add_module(
            f'Linear {len(layer_size)-1}', nn.Linear(layer_size[-2], layer_size[-1], use_bias))
        self.mlp.add_module(f'{out_activation}',
                            activation_dict[out_activation])

    def forward(self, x):
        x = self.mlp(x)
        return x


# 计算准确率
def compute_accuracy(model, loader, device):
    correct, total = 0, 0
    # 使用 inference_mode 进行推理
    with torch.inference_mode():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total


if __name__ == '__main__':
    # 超参数
    num_epochs = 1000
    learning_rate = 0.05
    batch_size = 128
    eps = 1e-7
    torch.manual_seed(0)

    # 使用GPU加速，如果可用的话
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 读入数据（异或数据集）
    data = np.loadtxt('xor_dataset.csv', delimiter=',')
    print('数据集大小：', data.shape)

    ratio = 0.8
    split = int(ratio*len(data))

    np.random.seed(0)
    data = np.random.permutation(data)
    x_train, y_train = data[:split, :2], data[:split, -1].reshape(-1, 1)
    x_test, y_test = data[split:, :2], data[split:, -1].reshape(-1, 1)

    # 将数据转换为PyTorch张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(
        x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        x_test_tensor, y_test_tensor), batch_size=batch_size)

    # 创建MLP模型并移动到GPU（如果有）
    mlp = MLP([2, 4, 1], use_bias=True).to(device)

    # 损失函数和优化器
    Loss_Func = nn.BCELoss()  # 二分类任务用二元交叉熵损失
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

    # 存储损失和准确率
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []

    # 训练模型
    for epoch in range(num_epochs):
        mlp.train()  # 模型处于训练模式
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(
                device), y_batch.to(device)  # 数据放到GPU
            # 前向传播
            outputs = mlp(x_batch)
            loss = Loss_Func(outputs, y_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算并记录训练集损失和准确率
        train_losses.append(running_loss / len(train_loader))
        train_acc.append(compute_accuracy(mlp, train_loader, device))

        # 在测试集上评估
        mlp.eval()  # 模型处于评估模式
        test_loss = 0.0
        with torch.inference_mode():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = mlp(x_batch)
                loss = Loss_Func(outputs, y_batch)
                test_loss += loss.item()

        # 记录测试损失
        test_losses.append(test_loss / len(test_loader))
        test_acc.append(compute_accuracy(mlp, test_loader, device))

        # 打印训练过程中的损失
        if (epoch + 1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印最终测试集准确率
    final_accuracy = compute_accuracy(mlp, test_loader, device)
    print(f'Final Test Accuracy: {final_accuracy * 100:.2f}%')
