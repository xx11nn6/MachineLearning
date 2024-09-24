import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from mnist_data_read import *


class KNN:
    def __init__(self, k, train_data, train_labels):
        self.k = k
        self.train_data = train_data
        self.labels = train_labels

    def get_knn_indices(self, x):
        # 获取x到最近k个样本的下标
        # 获取x到所有图像的距离
        x_flat = x.flatten()  # 将x展开为一维向量
        data_flat = self.train_data.reshape(
            self.train_data.shape[0], -1)  # 将训练集每个图像展开为一维向量
        # 计算L2距离 (欧氏距离)
        dists = np.linalg.norm(data_flat - x_flat, axis=1)
        # 对距离进行从小到大的排序，取前k个
        sorted_indices = np.argsort(dists)[:self.k]
        # print(sorted_indices)
        return sorted_indices

    def get_label(self, x):
        # KNN实现单个数据的预测：对于前k个下标，选取数量最多的类别作为目标类别
        sorted_indices = self.get_knn_indices(x)
        labels = self.labels[sorted_indices]
        # print(labels)
        # 统计数组内的唯一值及出现频率
        unique, counts = np.unique(labels, return_counts=True)
        max_label = np.argmax(counts)  # 选取计数次数最多的下标
        predicted_label = int(unique[max_label])
        return predicted_label

    def predict(self, x_test):
        # 预测大批样本的类别
        x_test_flat = x_test.reshape(
            x_test.shape[0], -1)  # 将测试集每个图像展开为一维向量
        predicted_test_labels = np.zeros(shape=[len(x_test)], dtype=int)

        for i, x in enumerate(x_test):
            predicted_test_labels[i] = self.get_label(x)
        return predicted_test_labels


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    test_images = test_images[:100, :, :]
    test_labels = test_labels[:100]

    knn = KNN(5, train_images, train_labels)
    predicted_test_labels = knn.predict(test_images)
    accuracy = np.mean(predicted_test_labels == test_labels)  # 准确率
    print(accuracy)
