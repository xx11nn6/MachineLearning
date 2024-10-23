# 使用Pytorch实现CIFAR10数据集的分类
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# 数据处理
data_path = './cifar10'
trainset = CIFAR10(root=data_path, train=True, download=True,
                   transform=transforms.ToTensor())
testset = CIFAR10(root=data_path, train=False, download=True,
                  transform=transforms.ToTensor())

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

print(f'训练集大小:{len(trainset)}')
print(f'测试集大小:{len(testset)}')
print(trainset[0][0].shape)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # 卷积层
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),  # 3*32*32 -> 32*32*32
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding=1),  # 32*32*32 -> 32*32*32
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # 32*32*32 -> 32*16*16
            nn.Dropout(0.25),  # 添加第一个 Dropout 层，丢弃率为 25%

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      padding=1),  # 32*16*16 -> 64*16*16
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      padding=1),  # 64*16*16 -> 64*16*16
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # 64*16*16 -> 64*8*8
            nn.Dropout(0.25)  # 添加第二个 Dropout 层，丢弃率为 25%
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),  # 将多维度特征图展平
            nn.Linear(64 * 8 * 8, 512),  # 64*8*8 展平 -> 512个节点
            nn.ReLU(),
            nn.Dropout(0.5),  # 在全连接层前添加一个 Dropout，丢弃率为 50%
            nn.Linear(512, num_classes)  # 512个节点 -> num_classes（10个类别）
        )

    def forward(self, x):
        x = self.cnn(x)  # 卷积部分
        x = self.fc(x)    # 全连接部分
        return x


# 定义训练和测试函数
def train(model, device, trainloader, optimizer, criterion, epoch,  train_acc_list):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    train_acc_list.append(accuracy)
    print(
        f'Epoch [{epoch + 1}] Loss: {running_loss / len(trainloader):.3f}, Accuracy: {accuracy:.2f}%')


def test(model, device, testloader, criterion,  test_acc_list, all_preds, all_labels):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 在测试过程中不计算梯度
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 收集所有预测和真实标签，用于混淆矩阵
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    test_acc_list.append(accuracy)
    print(
        f'Test Loss: {running_loss / len(testloader):.3f}, Accuracy: {accuracy:.2f}%')


# 主程序
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=10).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数自动包含softmax
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化

# 存储训练和测试集的准确率
train_acc_list = []
test_acc_list = []

# 存储所有预测值和真实标签，用于混淆矩阵
all_preds = []
all_labels = []

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    train(model, device, trainloader, optimizer,
          criterion, epoch, train_acc_list)
    test(model, device, testloader, criterion,
         test_acc_list, all_preds, all_labels)

print("训练结束")

# 绘制训练和测试集的准确率变化图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_acc_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy Over Epochs')
plt.legend()
plt.grid()
plt.show()

# 计算并绘制混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
