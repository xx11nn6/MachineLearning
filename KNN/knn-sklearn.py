from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mnist_data_read import *

train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

train_images = train_images.reshape(
    train_images.shape[0], -1)  # 将训练集每个图像展开为一维向量

test_images = test_images.reshape(
    test_images.shape[0], -1)

# 直接用训练集数据分割，20%为测试集
# train_images, test_images, labels_train, labels_test = train_test_split(
#     data_flat, train_labels, test_size=0.2, random_state=42)

# 将训练集标准化（0均值，1方差）
scaler = StandardScaler()  # 创立scaler对象，用于均值化数据
scaler.fit(train_images)
train_images = scaler.transform(train_images)
test_images = scaler.transform(test_images)

# 初始化KNN分类器
K = 5
knn = KNeighborsClassifier(n_neighbors=K)

# 训练模型
knn.fit(train_images, train_labels)
# 预测测试集
labels_pred = knn.predict(test_images)

# 分类报告
print(classification_report(test_labels, labels_pred))

# 可视化混淆矩阵
confusion = confusion_matrix(test_labels, labels_pred)
plt.matshow(confusion)
# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 在每个单元格中添加数字
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        plt.text(x=j, y=i, s=str(confusion[i, j]),
                 va='center', ha='center', color='red')

plt.colorbar()
plt.ylabel('实际类型')
plt.xlabel('预测类型')
plt.title('混淆矩阵')
plt.show()
