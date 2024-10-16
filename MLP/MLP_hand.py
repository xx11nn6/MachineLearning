import numpy as np
import matplotlib.pyplot as plt

# 首先定义MLP的网络模型和一些激活函数的类
# 定义层的抽象结构


class Layer:
    # 前向传播函数，计算输出
    def forward(self, x):
        raise NotImplementedError  # 抽象类暂不实现，在子类中实现

    # 反向传播函数，输入上一层传回的梯度，输出当前层的梯度
    def backward(self, grad):
        raise NotImplementedError  # 暂不实现

    # 更新函数，更新当前层的参数
    def update(self, learning_rate):
        pass  # 暂不实现

# 线性层（全连接层），继承自Layer类


class Linear(Layer):
    def __init__(self, dim_in, dim_out, use_bias=True):
        self.dim_in = dim_in  # 输入维度
        self.dim_out = dim_out  # 输出维度
        self.use_bias = use_bias  # 是否添加偏置

        # 初始化参数，使用正态分布
        self.W = np.random.normal(loc=0, scale=1.0, size=(dim_in, dim_out))
        if use_bias:
            self.b = np.zeros((1, dim_out))

    def forward(self, x):
        # 前向传播y=Wx+b
        # x维度：batch_size * dim_in
        self.x = x
        self.y = x@self.W
        if self.use_bias:
            self.y += self.b
        # print(f'线性层权重矩阵维度为：{self.W.shape},输入x维度为：{x.shape},输出y维度为：{self.y.shape}')
        return self.y

    def backward(self, grad):
        # 反向传播，输入回传的梯度
        # 回传梯度，大小为batch_size * dim_out
        # 上层梯度为grad，当前层W的梯度为x，b的梯度为1
        # 则当前W的梯度为grad*x，b的梯度为grad
        self.grad_W = self.x.T @ grad/grad.shape[0]
        if self.use_bias:
            self.grad_b = np.mean(grad, axis=0, keepdims=True)
        # 输入x，输出y，则回传梯度为dy/dx=W，再乘以后一层回传的梯度grad
        grad_pre = grad @ self.W.T
        return grad_pre

    def update(self, learning_rate):
        self.W -= learning_rate*self.grad_W
        if self.use_bias:
            self.b -= learning_rate*self.grad_b


# 以下定义不同激活函数
class Identity(Layer):
    # 单位函数，输入=输出
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad


class Sigmoid(Layer):
    # sigmoid函数
    def forward(self, x):
        self.x = x
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, grad):
        return grad*self.y*(1-self.y)


class Tanh(Layer):
    # tanh函数
    def forward(self, x):
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad*(1-self.y**2)


class ReLU(Layer):
    # 唯一真神
    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y

    def backward(self, grad):
        return grad*(self.x >= 0)


# 存储激活函数到字典，便于索引
activation_dict = {
    'identity': Identity,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU
}

# MLP


class MLP:
    def __init__(self,
                 layer_sizes,  # 层数，包含每层大小，例如[4,5,5,2]输入特征维度为4，每层输出维度为5,5,2
                 use_bias=True,
                 activation='relu',
                 out_activation='identity'
                 ):
        self.layers = []
        dim_in = layer_sizes[0]
        for dim_out in layer_sizes[1:-1]:
            # 全连接层
            self.layers.append(Linear(dim_in, dim_out))
            # 激活函数
            self.layers.append(activation_dict[activation]())
            dim_in = dim_out
        # 如果输出需要特殊处理（如分类等），最后一层通常特殊处理
        self.layers.append(Linear(dim_in, layer_sizes[-1], use_bias))
        self.layers.append(activation_dict[out_activation]())

    def forward(self, x):
        # 前向传播
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        # 反向传播，梯度反向计算
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        # 更新每一层参数
        for layer in self.layers:
            layer.update(learning_rate)


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

    # 使用MLP训练二分类问题
    # 设置超参数
    num_epochs = 1000
    learning_rate = 0.1
    batch_size = 128
    eps = 1e-7  # 防止除以0，log(0)等问题

    # 创建一个[2,4,1]的MLP
    mlp = MLP(layer_sizes=[2, 4, 4, 1], use_bias=True,
              activation='relu', out_activation='sigmoid')

    # 训练
    losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        # SGD
        start = 0
        loss = 0.0
        while True:
            end = min(start+batch_size, len(x_train))
            if start >= end:
                break
            # 取出batch
            x = x_train[start:end]
            y = y_train[start:end]

            y_pred = mlp.forward(x)
            # 计算梯度(交叉熵梯度)
            grad = (y_pred-y)/(y_pred*(1-y_pred)+eps)
            # 反向传播
            mlp.backward(grad)
            mlp.update(learning_rate)
            # 计算交叉熵损失
            train_loss = np.sum(-y*np.log(np.clip(y_pred, eps, 1-eps))
                                - (1-y)*np.log(np.clip(1-y_pred, eps, 1-eps)))
            loss += train_loss
            start += batch_size

        losses.append(loss/len(x_train))
        # 测试集验证
        y_pred = mlp.forward(x_test)
        # print(y_pred.shape)
        test_loss = np.sum(-y_test*np.log(np.clip(y_pred, eps, 1-eps))
                           - (1-y_test)*np.log(np.clip(1-y_pred, eps, 1-eps)))/len(x_test)
        test_accuracy = np.sum(np.round(y_pred) == y_test)/len(x_test)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

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
