import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 读入数据
lines = np.loadtxt('lr_dataset.csv', delimiter=',', dtype=float)
x_total = lines[:, 0:2]
y_total = lines[:, 2]
print('数据集大小:', len(x_total))

# 绘图
pos_index = np.where(y_total == 1)
neg_index = np.where(y_total == 0)
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1],
            marker='o', color='coral', s=10)
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1],
            marker='x', color='blue', s=10)
plt.xlabel('X1 axis')
plt.ylabel('X2 axis')
plt.show()

# 划分测试集与训练集
np.random.seed(0)
ratio = 0.7
split = int(len(x_total)*ratio)
idx = np.random.permutation(len(x_total))  # 打乱数据集
x_total = x_total[idx]
y_total = y_total[idx]
x_train, y_train = x_total[:split], y_total[:split]
x_test, y_test = x_total[split:], y_total[split:]

X = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

# 准确率


def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)

# 曲线下面积


def auc(y_true, y_pred):
    # 按预测值从大到小排序
    idx = np.argsort(y_pred)[::1]
    y_true = y_true[idx]
    tp = np.cumsum(y_true)  # 真阳性个数
    fp = np.cumsum(1-y_true)  # 假阳性个数
    tpr = tp/tp[-1]
    fpr = fp/fp[-1]

    s = 0
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    for i in range(1, len(fpr)):
        s += (fpr[i]-fpr[i-1])*tpr[i]
    return s

# sigmoid函数


def sigmoid(z):
    return 1/(1+np.exp(-z))

# 梯度下降


def GD(num_steps, learning_rate, l2_coef):
    # 初始化
    theta = np.random.normal(size=(X.shape[1],))

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []
    for i in range(num_steps):
        pred = sigmoid(X@theta)
        grad = -X.T@(y_train-pred)+l2_coef*theta  # σ(Xθ)的梯度，加上L2正则化
        theta -= learning_rate*grad
        # 记录损失函数
        train_loss = - \
            y_train.T@np.log(pred)-(1-y_train).T@np.log(1-pred) + \
            l2_coef*np.linalg.norm(theta)**2/2
        train_losses.append(train_loss/len(X))

        test_pred = sigmoid(X_test@theta)
        test_loss = - \
            y_test.T@np.log(test_pred)-(1-y_test).T@np.log(1-test_pred)
        test_losses.append(test_loss/len(X_test))

        # 记录指标
        train_acc.append(acc(y_train, pred >= 0.5))
        test_acc.append(acc(y_test, test_pred >= 0.5))
        train_auc.append(auc(y_train, pred))
        test_auc.append(auc(y_test, test_pred))

    return theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc


# 定义超参数
num_steps = 250
learning_rate = 0.002
l2_coef = 1
np.random.seed(0)

theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc = GD(
    num_steps, learning_rate, l2_coef)

# 计算准确率
y_pred = np.where(sigmoid(X_test@theta) >= 0.5, 1, 0)
print('准确率：', acc(y_test, y_pred))
print('回归系数：', theta)

# 绘制训练曲线
plt.figure(figsize=(13, 9))
xticks = np.arange(num_steps)+1

plt.subplot(221)
plt.plot(xticks, train_losses, color='blue', label='train_loss')
plt.plot(xticks, test_losses, color='red', label='test_loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率
plt.subplot(222)
plt.plot(xticks, train_acc, color='blue', label='train_accuracy')
plt.plot(xticks, test_acc, color='red', label='test_accuracy')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制AUC
plt.subplot(223)
plt.plot(xticks, train_auc, color='blue', label='train_AUC')
plt.plot(xticks, test_auc, color='red', label='test_AUC')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

plt.show()
