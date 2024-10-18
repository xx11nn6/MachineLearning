# pytorch实现MLP,使用torch已定义的网络结构、每一层与优化算法
# 但损失函数、训练过程仍然手动实现

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_  # 正态分布初始化
import matplotlib.pyplot as plt

activation_dict = {
    'identity': lambda x: x,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'relu': torch.relu
}

# 定义MLP，继承自nn.Module
# 只需实现前向传播，后向传播和梯度由torch实现


class MLP_torch(nn.Module):
    def __init__(self, layer_sizes,
                 use_bias=True,
                 activation='relu',
                 out_activation='identiy'):
        super().__init__()  # 初始化父类
        self.activation = activation_dict[activation]
        self.out_activation = activation_dict[out_activation]
        self.layers = nn.ModuleList()  # ModuleList以列表方式存储
        dim_in = layer_sizes[0]
        for dim_out in layer_sizes[1:]:
            # 创建全连接层
            self.layers.append(nn.Linear(dim_in, dim_out, bias=use_bias))
            # 初始化全连接权重
            normal_(self.layers[-1].weight, std=1.0)
            # 初始偏置项为0
            self.layers[-1].bias.data.fill_(0.0)
            dim_in = dim_out

    def forward(Self, x):
        # 前向传播
        # pytorch可以自动处理batch_size等维度问题
        for i in range(len(Self.layers)-1):
            x = Self.layers[i](x)
            x = Self.activation(x)
        # 输出
        x = Self.layers[-1](x)
        x = Self.out_activation(x)
        return x


if __name__ == '__main__':
    # 读入数据（异或数据集）
    data = np.loadtxt('xor_dataset.csv', delimiter=',')
    print('数据集大小：', len(data))

    ratio = 0.8
    split = int(ratio*len(data))

    np.random.seed(0)
    data = np.random.permutation(data)
    x_train, y_train = data[:split, :2], data[:split, -1].reshape(-1, 1)
    x_test, y_test = data[split:, :2], data[split:, -1].reshape(-1, 1)

    # 超参数
    num_epochs = 1000
    learning_rate = 0.1
    batch_size = 128
    eps = 1e-7
    torch.manual_seed(0)

    mlp = MLP_torch(layer_sizes=[2, 4, 1],
                    use_bias=True, out_activation='sigmoid')
    # 定义优化器
    opt = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

    losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        start = 0
        loss = []
        while True:
            end = min(start+batch_size, len(x_train))
            if start >= end:
                break
            # 取出batch转化为张量
            x = torch.tensor(x_train[start:end], dtype=torch.float32)
            y = torch.tensor(y_train[start:end],
                             dtype=torch.float32).reshape(-1, 1)
            # torch自动调用前向传播
            y_pred = mlp(x)
            # 计算交叉熵
            train_loss = torch.mean(-y*torch.log(y_pred+eps) -
                                    (1-y)*torch.log(1-y_pred+eps))
            # 清空梯度
            opt.zero_grad()
            # 反向传播
            train_loss.backward()
            opt.step()

            loss.append(train_loss.detach().numpy())
            start += batch_size

        losses.append(np.mean(loss))
        # 计算测试集交叉熵，使用torch.inference_mode加速
        with torch.inference_mode():
            x = torch.tensor(x_test, dtype=torch.float32)
            y = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
            y_pred = mlp(x)
            test_loss = torch.sum(-y*torch.log(y_pred+eps) -
                                  (1-y)*torch.log(1-y_pred+eps))/len(x_test)
            test_acc = torch.sum(torch.round(y_pred) == y)/len(x_test)
            test_losses.append(test_loss.detach().numpy())
            test_accuracies.append(test_acc)

    print('测试准确率：', test_accuracies[-1])
    # 作图
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(losses, color='blue', label='train loss')
    plt.plot(test_losses, color='red', label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Cross-Entropy Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(test_accuracies, color='red')
    plt.ylim(top=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.show()
