import numpy as np
import matplotlib.pyplot as plt

# 读入数据（异或数据集）
data = np.loadtxt('xor_dataset.csv', delimiter=',')
print('数据集大小：', len(data))

ratio = 0.8
split = int(ratio*len(data))

np.random.seed(0)
data = np.random.permutation(data)
x_train, y_train = data[:split, :2], data[:split, -1].reshape(-1, 1)
x_test, y_test = data[split:, :2], data[split:, -1].reshape(-1, 1)

# 定义每一层结构


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
        self.W = np.random.normal(loc=0, scale=1.0, size=(dim_out, dim_in))
        if use_bias:
            self.b = np.zeros((1, dim_out))

    def forward(self, x):
        # 前向传播y=Wx+b
        # x维度：batch_size * dim_in
        self.x = x
        self.y = x@self.W
        if self.use_bias:
            self.y += self.b
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
