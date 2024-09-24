from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from mnist_data_read import *

train_images = load_train_images()
train_labels = load_train_labels()

# 直接用训练集数据分割，20%为测试集
images_train, images_test, labels_train, labels_test = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42)
