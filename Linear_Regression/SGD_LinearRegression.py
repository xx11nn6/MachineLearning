import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator
# 生成随机小批量样本


def batch_generator(x, y, batch_size, shuffle=True):
    # 输入，标签，样本量，是否重新划分
    count = 0
    if shuffle:
        idx = np.random.permutation(len(x))  # 按照给定列表生成一个打乱后的随机列表
        x = x[idx]
        y = y[idx]

    while True:
        start = count*batch_size
        end = min(start+batch_size, len(x))
        if start >= end:
            # 已经遍历一遍，结束
            break
        count += 1
        yield x[start:end], y[start:end]  # 生成器，每次调用选取batch_size的数据


def SGD_LiReg(x_train, y_train, x_test, y_test, num_epoch, learning_rate, batch_size):
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)  # 添加偏置项
    X_test = np.concatenate(
        [x_test, np.ones((len(x_test), 1))], axis=-1)  # 添加偏置项
    theta = np.random.normal(size=X.shape[1])

    train_losses = []
    test_losses = []
    for i in range(num_epoch):
        batch = batch_generator(X, y_train, batch_size)
        train_loss = 0
        for x_batch, y_batch in batch:
            # 计算梯度
            predictions = x_batch @ theta  # 预测值
            grad = x_batch.T @ (predictions - y_batch)  # 线性回归的梯度
            # 更新参数
            theta -= learning_rate * grad / len(x_batch)
            # 计算损失
            train_loss += np.square(predictions - y_batch).sum()

        # 计算训练误差
        train_loss = np.sqrt(train_loss / len(X))
        train_losses.append(train_loss)
        test_loss = np.sqrt(np.square(X_test@theta-y_test).mean())
        test_losses.append(test_loss)

    bias = theta[-1]
    theta = theta[:-1]

    return theta, bias, train_losses, test_losses


# 从文件按行加载数据
lines = np.loadtxt('USA_Housing.csv', delimiter=',', dtype='str')
header = lines[0]
lines = lines[1:].astype(float)
print('特征：', ','.join(header[:-1]))
print('标签：', header[-1])
print('数据大小：', lines.shape)

# 划分训练集与测试集
ratio = 0.8
split = int(len(lines)*ratio)
lines = np.random.permutation(lines)
train, test = lines[:split], lines[split:]

# 数据标准化
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# 划分输入与标签
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()

# 设置超参数
num_epoch = 20
learning_rate = 0.01
batch_size = 32

# 训练
theta, bias, train_losses, test_losses = SGD_LiReg(
    x_train, y_train, x_test, y_test, num_epoch, learning_rate, batch_size)

# 绘制训练轮数和误差关系图
plt.plot(np.arange(num_epoch), train_losses, color='blue', label='train loss')
plt.plot(np.arange(num_epoch), test_losses,
         color='red', ls='--', label='test loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置坐标为整数
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()
