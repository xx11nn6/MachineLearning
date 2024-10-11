# 矩阵分解实现推荐系统
# 通过用户对少量电影的评分，预测用户可能喜欢的电影
# 模型假设电影有d个特征，n个用户的偏好形成矩阵P(n*d)，m个电影的特征矩阵Q(m*d)，用户的评分矩阵T=PQ'(n*m)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条

data = np.loadtxt('movielens_100k.csv', delimiter=',',
                  dtype=int)  # 读入电影评分数据，三个维度为用户id，电影id，评分
print('数据集大小：', len(data))
data[:, :2] = data[:, :2]-1  # id从1开始，转化为从0开始

users = set()  # 无序集合统计用户量
items = set()
for i, j, k in data:
    users.add(i)
    items.add(j)

user_num = len(users)
item_num = len(items)
print(f'用户数:{user_num}，电影数：{item_num}')

# 划分训练集与测试集
np.random.seed(0)
ratio = 0.8
split = int(len(data)*ratio)
np.random.shuffle(data)
train = data[:split]
test = data[split:]

# 统计训练集用户与电影出现数量，作为正则化约束强度
# 从0到minlength，每个元素出现的次数
user_count = np.bincount(train[:, 0], minlength=user_num)
item_count = np.bincount(train[:, 1], minlength=item_num)
print(user_count[:10])
print(item_count[:10])

user_train, user_test = train[:, 0], test[:, 0]
item_train, item_test = train[:, 1], test[:, 1]
y_train, y_test = train[:, 2], test[:, 2]

# 定义MF


class MF:
    def __init__(self, N, M, d):
        # 初始化参数矩阵，随机生成参数
        self.user_params = np.ones((N, d))
        self.item_params = np.ones((M, d))

    def pred(self, user_id, item_id):
        # 预测用户对电影的评分
        user_param = self.user_params[user_id]
        item_param = self.item_params[item_id]

        rating_pred = np.sum(user_param*item_param, axis=1)
        return rating_pred

    def update(self, user_grad, item_grad, lr):
        # 根据梯度更新参数
        self.user_params -= lr*user_grad
        self.item_params -= lr*item_grad


def train(model, learning_rate, l_coef, max_training_step, batch_size):
    train_losses = []
    test_losses = []
    batch_num = int(np.ceil(len(user_train)/batch_size))
    with tqdm(range(max_training_step)) as pbar:  # 每个 epoch 进度条
        for epoch in range(max_training_step):
            # mini-batch SGD
            train_rmse = 0
            for i in range(batch_num):
                start = i * batch_size
                end = min(len(user_train), start + batch_size)
                user_batch = user_train[start:end]
                item_batch = item_train[start:end]
                y_batch = y_train[start:end]

                y_pred = model.pred(user_batch, item_batch)
                # 计算梯度
                P = model.user_params
                Q = model.item_params
                errors = y_batch - y_pred
                P_grad = np.zeros_like(P)
                Q_grad = np.zeros_like(Q)
                for user, item, error in zip(user_batch, item_batch, errors):
                    P_grad[user] = P_grad[user] - \
                        error * Q[item] + l_coef * P[user]
                    Q_grad[item] = Q_grad[item] - \
                        error * P[user] + l_coef * Q[item]

                model.update(P_grad / len(user_batch), Q_grad /
                             len(user_batch), learning_rate)

                train_rmse += np.mean(errors ** 2)

            # 计算每个 epoch 的训练集 RMSE
            train_rmse = np.sqrt(train_rmse / len(user_train))
            train_losses.append(train_rmse)

            # 计算每个 epoch 的测试集 RMSE
            y_test_pred = model.pred(user_test, item_test)
            test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            test_losses.append(test_rmse)

            # 更新进度条
            pbar.set_postfix({
                'epoch': epoch,
                'Train RMSE': f'{train_rmse:.4f}',
                'Test RMSE': f'{test_rmse:.4f}'
            })
            pbar.update(1)

    return train_losses, test_losses


# 定义超参数
feature_num = 16  # 特征个数
learning_rate = 0.1
l_coef = 1e-4
max_training_step = 30
batch_size = 64

# 建立模型
model = MF(user_num, item_num, feature_num)
train_losses, test_losses = train(model,
                                  learning_rate, l_coef, max_training_step, batch_size)

plt.figure()
x = np.arange(max_training_step)+1
plt.plot(x, train_losses, color='blue', label='train loss')
plt.plot(x, test_losses, color='red', ls='--', label='test loss')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()
